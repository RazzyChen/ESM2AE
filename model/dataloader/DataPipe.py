import json
import pickle
from typing import Any, Dict, Iterator, List

import lmdb
from datasets import Dataset
from transformers import AutoTokenizer


def _dataset_generator(mdb_file_path: str) -> Iterator[Dict[str, Any]]:
    """
    A generator that yields data entries from an LMDB file.
    This avoids loading the entire dataset into memory.
    """
    with lmdb.open(
        mdb_file_path, readonly=True, lock=False, readahead=False, meminit=False
    ) as env:
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                try:
                    try:
                        # First, try to decode as JSON
                        entry = json.loads(value.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        # If JSON decoding fails, try to deserialize with pickle
                        entry = pickle.loads(value)

                    # Check if entry is a dictionary and has the required fields
                    if (
                        isinstance(entry, dict)
                        and "sequence" in entry
                        and isinstance(entry.get("sequence"), str)
                    ):
                        yield {"sequence": entry["sequence"]}

                except Exception as e:
                    print(
                        f"Skipping invalid entry with key {key.decode('ascii', errors='ignore')}: {e}"
                    )
                    continue


def load_and_preprocess_data(train_mdb_path: str, tokenizer: AutoTokenizer) -> Dataset:
    """
    Load and preprocess the training dataset from an LMDB file in a memory-efficient way.

    Args:
        train_mdb_path (str): Path to the LMDB file containing training JSON data.
        tokenizer (AutoTokenizer): Tokenizer for preprocessing sequences.

    Returns:
        Dataset: A tokenized training dataset.
    """
    # Create a Hugging Face Dataset from the generator.
    # This streams the data instead of loading it all into memory.
    train_dataset = Dataset.from_generator(
        _dataset_generator,
        gen_kwargs={"mdb_file_path": train_mdb_path},
    )

    # Preprocessing function
    def preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, List[Any]]:
        sequences = [" " + seq for seq in examples["sequence"]]
        tokenized = tokenizer(
            sequences,
            padding="max_length",
            truncation=True,
            max_length=700,
        )
        return tokenized

    # Process the dataset using map. This is still efficient as `datasets`
    # processes it in batches.
    tokenized_train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    return tokenized_train_dataset
