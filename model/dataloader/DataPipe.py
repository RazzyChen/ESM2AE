import json
import pickle
from typing import Any, Dict, List

import lmdb
from datasets import Dataset
from transformers import AutoTokenizer


def load_and_preprocess_data(train_mdb_path: str, tokenizer: AutoTokenizer) -> Dataset:
    """
    Load and preprocess the training dataset from an LMDB file.

    Args:
        train_mdb_path (str): Path to the LMDB file containing training JSON data.
        tokenizer (AutoTokenizer): Tokenizer for preprocessing sequences.

    Returns:
        Dataset: A tokenized training dataset.
    """

    def load_lmdb_data(mdb_file_path: str) -> List[Dict[str, Any]]:
        """
        Helper function to load data from an LMDB file.
        """
        data = []
        with lmdb.open(
            mdb_file_path, readonly=True, lock=False, readahead=False, meminit=False
        ) as env:
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    try:
                        # First, try to decode as JSON
                        entry = json.loads(value.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        try:
                            # If JSON decoding fails, try to deserialize with pickle
                            entry = pickle.loads(value)
                        except Exception as e:
                            print(
                                f"Skipping invalid entry with key {key.decode('ascii')}: {e}"
                            )
                            continue

                    # Check if entry is a dictionary before checking for keys
                    if not isinstance(entry, dict):
                        print(
                            f"Skipping invalid entry with key {key.decode('ascii')}: Entry is not a dictionary but {type(entry)}"
                        )
                        continue

                    # Ensure the required field exists
                    if "sequence" not in entry:
                        print(
                            f"Skipping invalid entry with key {key.decode('ascii')}: Missing 'sequence'"
                        )
                        continue

                    # Validate field type
                    if not isinstance(entry.get("sequence"), str):
                        print(
                            f"Skipping invalid entry with key {key.decode('ascii')}: Invalid 'sequence'"
                        )
                        continue

                    data.append(entry)
        return data

    # Load training data from the LMDB path
    train_data = load_lmdb_data(train_mdb_path)

    # Convert the loaded data into DataFrame-like structures
    train_df = {
        "sequence": [item["sequence"] for item in train_data],
    }

    # Convert to Hugging Face Datasets
    train_dataset = Dataset.from_dict(train_df)

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

    # Process the dataset
    tokenized_train_dataset = train_dataset.map(
        preprocess_function, batched=True, remove_columns=train_dataset.column_names
    )

    return tokenized_train_dataset
