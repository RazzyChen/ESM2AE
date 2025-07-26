import os
import shutil
import tempfile
import torch
import webdataset as wds
from transformers import AutoTokenizer, DataCollatorWithPadding

from model.dataloader.DataPipe import load_and_preprocess_data

# Helper function to create a dummy WebDataset
def create_dummy_webdataset(directory, num_shards, samples_per_shard):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    for i in range(num_shards):
        shard_path = os.path.join(directory, f"shard-{i:06d}.tar")
        with wds.ShardWriter(shard_path, maxcount=samples_per_shard) as sink:
            for j in range(samples_per_shard):
                # Create sequences of varying lengths
                seq_len = 50 + (i * 10) + j
                sequence = "A" * seq_len
                key = f"{i:06d}{j:06d}"
                sink.write({
                    "__key__": key,
                    "fasta": sequence.encode("utf-8"),
                    "header": f"id_{key}".encode("utf-8")
                })
    return directory

class TestDataLoading:
    def setup_method(self):
        """Set up a temporary directory and a dummy dataset for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.webdataset_path = create_dummy_webdataset(os.path.join(self.test_dir, "wds"), num_shards=2, samples_per_shard=5)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def teardown_method(self):
        """Clean up the temporary directory after tests."""
        shutil.rmtree(self.test_dir)

    def test_data_loading_and_bucketing(self):
        """Test that the WebDataset is loaded correctly with bucketing and dynamic padding."""
        batch_size = 4
        # Load the dataset using our pipeline
        dataset = load_and_preprocess_data(
            train_webdataset_path=self.webdataset_path,
            tokenizer=self.tokenizer,
            batch_size=batch_size
        )

        # Create a data collator for dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Create a PyTorch DataLoader
        # Note: For IterableDataset, batching is often handled within the dataset itself
        # or by a custom batching mechanism. Here we will manually batch for testing.
        
        iterator = iter(dataset)
        # Collect a few samples to form a batch
        samples = [next(iterator) for _ in range(batch_size)]

        # Use the data collator to create a batch with padding
        batch = data_collator(samples)

        # --- Assertions ---
        assert "input_ids" in batch
        assert "attention_mask" in batch

        # Check batch shape
        input_ids = batch["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        assert input_ids.shape[0] == batch_size

        # Check that padding has been applied
        # The length of the tensor should be the max length in the original sample list
        max_len_in_batch = max(len(s['input_ids']) for s in samples)
        assert input_ids.shape[1] == max_len_in_batch

        # Check attention mask validity
        attention_mask = batch["attention_mask"]
        for i in range(batch_size):
            original_len = len(samples[i]["input_ids"])
            # The mask should be 1 for real tokens and 0 for padding tokens
            assert attention_mask[i, :original_len].all() == 1
            if original_len < max_len_in_batch:
                assert attention_mask[i, original_len:].all() == 0
