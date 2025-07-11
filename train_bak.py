import hydra
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    EsmConfig,
    Trainer,
    TrainingArguments,
)

import wandb
from model.backbone.esm2ae import ESM2AE
from model.dataloader.DataPipe import load_and_preprocess_data
from model.utils.ModelSave import save_model
from model.utils.MyLRCallback import LogLearningRateCallback


@hydra.main(
    version_base=None, config_path="./train_config", config_name="trainer_config"
)
def main(cfg: DictConfig):
    # Define paths
    train_mdb_path = "./dataset/train_dataset"
    # Ensure the train_mdb_path exists
    import os

    if not os.path.exists(train_mdb_path):
        raise FileNotFoundError(
            f"Training dataset path {train_mdb_path} does not exist."
        )

    # Initialize accelerator
    accelerator = Accelerator()

    # Initialize wandb
    if accelerator.is_main_process:
        wandb.init(
            project=cfg.wandb.project,  # Change this to your project name
            resume=cfg.wandb.resume,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    accelerator.wait_for_everyone()
    # Print configuration
    if accelerator.is_main_process:
        print("Configuration:")
        print(OmegaConf.to_yaml(cfg))
    #         print("Accelerator state:", accelerator.state)
    print("Accelerator state:", accelerator.state, flush=True)

    # Load configuration and tokenizer
    config = EsmConfig.from_pretrained(
        "facebook/esm2_t33_650M_UR50D",
        position_embedding_type="rotary",
        num_labels=1,
        problem_type="regression",
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Initialize model
    model = ESM2AE(config)

    # Load and preprocess data
    tokenized_train_dataset = load_and_preprocess_data(train_mdb_path, tokenizer)

    # Train the model
    trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        tokenized_train_datasets=tokenized_train_dataset,
        cfg=cfg,
    )

    # Save the model
    if accelerator.is_main_process:
        save_dir = "./saved_model"
        save_model(trainer, save_dir)
        wandb.finish()


def train_model(model, tokenizer, tokenized_train_datasets, cfg=None):
    training_args = TrainingArguments(
        output_dir=cfg.trainer.output_dir if cfg is not None else "./checkpoint",
        save_strategy=cfg.trainer.save_strategy if cfg is not None else "epoch",
        learning_rate=cfg.trainer.learning_rate if cfg is not None else 5e-5,
        per_device_train_batch_size=cfg.trainer.per_device_train_batch_size
        if cfg is not None
        else 16,
        per_device_eval_batch_size=cfg.trainer.per_device_eval_batch_size
        if cfg is not None
        else 16,
        num_train_epochs=cfg.trainer.num_train_epochs if cfg is not None else 100,
        seed=cfg.trainer.seed if cfg is not None else 42,
        data_seed=cfg.trainer.data_seed if cfg is not None else 42,
        dataloader_num_workers=cfg.trainer.dataloader_num_workers
        if cfg is not None
        else 12,
        dataloader_prefetch_factor=cfg.trainer.dataloader_prefetch_factor
        if cfg is not None
        else 100,
        logging_dir=cfg.trainer.logging_dir if cfg is not None else "./logs",
        tf32=cfg.trainer.tf32 if cfg is not None else True,
        bf16=cfg.trainer.bf16 if cfg is not None else True,
        push_to_hub=cfg.trainer.push_to_hub if cfg is not None else False,
        report_to=cfg.trainer.report_to if cfg is not None else "wandb",
        weight_decay=cfg.trainer.weight_decay if cfg is not None else 1e-4,
        adam_beta2=cfg.trainer.adam_beta2 if cfg is not None else 0.95,
        save_safetensors=cfg.trainer.save_safetensors if cfg is not None else True,
        greater_is_better=cfg.trainer.greater_is_better if cfg is not None else False,
        load_best_model_at_end=cfg.trainer.load_best_model_at_end
        if cfg is not None
        else False,
        optim=cfg.trainer.optim if cfg is not None else "adamw_torch_fused",
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps
        if cfg is not None
        else 1,
        metric_for_best_model=cfg.trainer.metric_for_best_model
        if cfg is not None
        else "mse",
        logging_steps=cfg.trainer.logging_steps if cfg is not None else 10,
        warmup_ratio=cfg.trainer.warmup_ratio if cfg is not None else 0.1,
        lr_scheduler_type=cfg.trainer.lr_scheduler_type
        if cfg is not None
        else "linear",
        save_total_limit=cfg.trainer.save_total_limit if cfg is not None else 10,
    )
    # Define early stopping callback (not needed for self-supervised, but kept for flexibility)
    callbacks = [LogLearningRateCallback()]

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_datasets,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
    # Start training
    trainer.train()
    return trainer


if __name__ == "__main__":
    main()
