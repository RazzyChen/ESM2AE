import json
import os
import pickle
import sys
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

import lmdb
import ray
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


@contextmanager
def suppress_output(is_main_worker):
    """A context manager that suppresses stdout and stderr for non-main workers."""
    if is_main_worker:
        yield
        return

    with open(os.devnull, "w") as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def _dataset_generator(mdb_file_path: str) -> Iterator[Dict[str, Any]]:
    """
    A generator that yields data entries from an LMDB file with a progress bar
    on the main worker.
    """
    try:
        rank = ray.train.get_context().get_world_rank()
    except (ValueError, AttributeError):
        rank = 0

    # Suppress all output from non-main workers
    with suppress_output(is_main_worker=(rank == 0)):
        with lmdb.open(
            mdb_file_path, readonly=True, lock=False, readahead=False, meminit=False
        ) as env:
            num_examples = env.stat()["entries"]
            with env.begin() as txn:
                cursor = txn.cursor()

                pbar = tqdm(
                    total=num_examples,
                    desc="Generating dataset",
                    disable=(rank != 0),
                )

                with pbar:
                    for key, value in cursor:
                        try:
                            try:
                                entry = json.loads(value.decode("utf-8"))
                            except (UnicodeDecodeError, json.JSONDecodeError):
                                entry = pickle.loads(value)

                            if (
                                isinstance(entry, dict)
                                and "sequence" in entry
                                and isinstance(entry.get("sequence"), str)
                            ):
                                yield {"sequence": entry["sequence"]}

                        except Exception as e:
                            if rank == 0:
                                print(
                                    f"Skipping invalid entry with key {key.decode('ascii', errors='ignore')}: {e}"
                                )
                        finally:
                            pbar.update(1)


def load_and_preprocess_data(
    train_mdb_path: str, tokenizer: AutoTokenizer, cache_dir: str
) -> Dataset:
    """
    Load and preprocess the training dataset from an LMDB file in a memory-efficient way.
    Caches the tokenized dataset to disk if a cache_dir is provided.
    """
    if os.path.exists(cache_dir):
        print(f"Loading tokenized dataset from cache: {cache_dir}")
        return Dataset.load_from_disk(cache_dir)

    try:
        rank = ray.train.get_context().get_world_rank()
    except (ValueError, AttributeError):
        rank = 0

    # Suppress output during dataset generation
    with suppress_output(is_main_worker=(rank == 0)):
        train_dataset = Dataset.from_generator(
            _dataset_generator,
            gen_kwargs={"mdb_file_path": train_mdb_path},
        )

    def preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, List[Any]]:
        sequences = [" " + seq for seq in examples["sequence"]]
        tokenized = tokenizer(
            sequences,
            padding="max_length",
            truncation=True,
            max_length=700,
        )
        return tokenized

    num_proc_to_use = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
    tokenized_train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing dataset",
        disable_nullable=(rank != 0),
        num_proc=num_proc_to_use,
    )

    if rank == 0:
        print(f"Saving tokenized dataset to cache: {cache_dir}")
        tokenized_train_dataset.save_to_disk(cache_dir)

    return tokenized_train_dataset
