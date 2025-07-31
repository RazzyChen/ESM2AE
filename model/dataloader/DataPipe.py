import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

import ray
import webdataset as wds
from torch.utils.data import IterableDataset
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


def preprocess_function(example: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Tokenize a single example *without* padding.
    Padding will be handled dynamically by the DataCollator.
    """
    # The sequence is expected to be in bytes, so we decode it first.
    sequence = " " + example["fasta"].decode("utf-8")
    # IMPORTANT: No padding here. `padding=False` is the default.
    tokenized = tokenizer(
        sequence,
        truncation=True,  # Still truncate to a max sensible length
        max_length=1024, # Increased max_length slightly
    )
    return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}


class WebDatasetIterable(IterableDataset):
    def __init__(self, urls: List[str], tokenizer: AutoTokenizer):
        super().__init__()
        self.urls = urls
        self.tokenizer = tokenizer

        try:
            self.rank = ray.train.get_context().get_world_rank()
        except (ValueError, AttributeError):
            self.rank = 0

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        with suppress_output(is_main_worker=(self.rank == 0)):
            dataset = wds.WebDataset(self.urls, nodesplitter=wds.split_by_node, shardshuffle=True)
            
            # Shuffle, decode, and tokenize
            dataset = dataset.shuffle(1000).decode()
            dataset = dataset.map(lambda x: preprocess_function(x, self.tokenizer))

            for item in dataset:
                yield item


def load_and_preprocess_data(
    train_webdataset_path: str, tokenizer: AutoTokenizer, **kwargs
) -> WebDatasetIterable:
    """
    Load and preprocess data using WebDataset.
    """
    if not os.path.isdir(train_webdataset_path):
        raise FileNotFoundError(
            f"WebDataset path not found or is not a directory: {train_webdataset_path}"
        )

    urls = [
        os.path.join(train_webdataset_path, f)
        for f in os.listdir(train_webdataset_path)
        if f.endswith(".tar")
    ]
    if not urls:
        raise FileNotFoundError(
            f"No .tar shards found in directory: {train_webdataset_path}"
        )
    
    print(f"Found {len(urls)} WebDataset shards. Initializing.")

    return WebDatasetIterable(urls, tokenizer)
