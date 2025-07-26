# ESM2AE: An Autoencoder for Protein Sequences based on ESM-2

ESM2AE is a self-supervised learning framework built upon the powerful ESM-2 protein language model. It learns compact and informative latent representations of protein sequences by training an autoencoder to reconstruct ESM-2's own representations.

## Key Features

- **ESM-2 Backbone**: Leverages a pre-trained ESM-2 model (e.g., `facebook/esm2_t6_8M_UR50D`) as a powerful feature extractor.
- **Autoencoder Architecture**: Compresses high-dimensional ESM-2 outputs into a low-dimensional latent space and reconstructs them, using an efficient architecture of Linear layers and SwiGLU activations.
- **Efficient Data Handling**: Utilizes **WebDataset** for true streaming of large datasets, eliminating the need to load everything into memory. It intelligently handles variable-length sequences by implementing **bucketing and dynamic padding**, which significantly reduces computational overhead from excessive padding.
- **Distributed Training**: Natively supports high-performance distributed training using **Ray Train** and is optimized for memory efficiency with **DeepSpeed (ZERO-2)**.
- **Self-Supervised**: The training is entirely self-supervised, optimized by minimizing the Mean Squared Error (MSE) between the reconstructed and original ESM-2 representations.

## Project Structure

```
ESM2AE/
├── model/                     # Model definitions
│   ├── backbone/              # Core model architecture (ESM2AE)
│   ├── dataloader/            # Data loading pipeline (WebDataset)
│   └── utils/                 # Helpers (Model Saving, Callbacks)
├── tools/                     # Data preprocessing scripts
│   ├── fasta2webdataset.py    # Converts FASTA to WebDataset format (Core)
│   ├── fasta_filter.py        # Filters FASTA by sequence length
│   ├── csv2fasta.py           # Converts CSV to FASTA
│   └── ...
├── train_config/              # Training configurations
│   ├── trainer_config.yaml    # Main training config (Hydra)
│   └── ZERO2.yaml             # DeepSpeed config
├── train_ray.py               # Main script for distributed training with Ray
└── README.md                  # This file
```

## Setup

It is recommended to use `uv` for fast dependency management.

```bash
# Install uv (if you haven't already)
# pip install uv

# Create a virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt
```

## Data Preparation

The training data must be converted to the WebDataset format for efficient streaming. This new format avoids the need for a large, single-file database and is ideal for cloud and distributed environments.

1.  **Prepare a FASTA file**:
    Ensure you have a single FASTA file containing all your protein sequences. You can use `tools/csv2fasta.py` or `tools/remove_duplicates.py` to clean your source data.

2.  **Convert FASTA to WebDataset**:
    This is the core preparation step. The `fasta2webdataset.py` script efficiently converts your FASTA file into a directory of `.tar` files (shards), which is the WebDataset format. This script uses a multi-process producer-consumer model to handle terabyte-scale files.

    ```bash
    # Define paths
    FASTA_FILE="/path/to/your/protein.fasta"
    OUTPUT_DIR="./data/protein_webdataset"

    # Create output directory
    mkdir -p $OUTPUT_DIR

    # Run the conversion script
    python tools/fasta2webdataset.py \
        --fasta_file $FASTA_FILE \
        --output_dir $OUTPUT_DIR \
        --processes 8  # Adjust based on your CPU cores
    ```
    After running, `$OUTPUT_DIR` will contain multiple `shard-xxxxxx.tar` files.

## Training

The project is configured for distributed training on a cluster using Ray, Hydra, and DeepSpeed.

### Configuration

-   **`train_config/trainer_config.yaml`**: This is the main configuration file. **Crucially, you must update the `data.train_path` to point to the directory containing your WebDataset shards** (e.g., `./data/protein_webdataset`). You can also configure hyperparameters like learning rate, batch size, and epochs here.
-   **`train_config/ZERO2.yaml`**: The DeepSpeed configuration for memory optimization.
-   **Ray Setup**: The number of workers for distributed training is set in `trainer_config.yaml` under the `ray` section.

### Launching the Training Job

With your configuration set, start the training by running the `train_ray.py` script. It will automatically initialize the Ray cluster and distribute the workload.

```bash
python train_ray.py
```

## Monitoring

Training progress is logged using `wandb`. Ensure you have set your project details in `train_config/trainer_config.yaml` to enable logging.

```