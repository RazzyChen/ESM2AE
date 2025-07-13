import os

import hydra
import ray
import ray.train.huggingface.transformers
import yaml
from omegaconf import DictConfig, OmegaConf
from ray.train import ScalingConfig
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


def train_func(config: dict):
    """Training function to be executed by each Ray worker."""
    cfg = OmegaConf.create(config)

    # Load DeepSpeed config
    with open(cfg.deepspeed.config_path, "r") as f:
        ds_config = yaml.safe_load(f)

    # Initialize wandb on the main worker
    if ray.train.get_context().get_world_rank() == 0:
        wandb.init(
            project=cfg.wandb.project,
            resume=cfg.wandb.resume,
            mode="offline",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

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
    train_mdb_path = cfg.data.train_path
    if not os.path.exists(train_mdb_path):
        raise FileNotFoundError(
            f"Training dataset path {train_mdb_path} does not exist."
        )
    tokenized_train_dataset = load_and_preprocess_data(train_mdb_path, tokenizer)

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir=cfg.trainer.output_dir,
        save_strategy=cfg.trainer.save_strategy,
        learning_rate=cfg.trainer.learning_rate,
        per_device_train_batch_size=cfg.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.trainer.per_device_eval_batch_size,
        num_train_epochs=cfg.trainer.num_train_epochs,
        seed=cfg.trainer.seed,
        data_seed=cfg.trainer.data_seed,
        dataloader_num_workers=cfg.trainer.dataloader_num_workers,
        dataloader_prefetch_factor=cfg.trainer.dataloader_prefetch_factor,
        logging_dir=cfg.trainer.logging_dir,
        tf32=cfg.trainer.tf32,
        bf16=cfg.trainer.bf16,
        push_to_hub=cfg.trainer.push_to_hub,
        report_to=cfg.trainer.report_to,
        weight_decay=cfg.trainer.weight_decay,
        adam_beta2=cfg.trainer.adam_beta2,
        save_safetensors=cfg.trainer.save_safetensors,
        greater_is_better=cfg.trainer.greater_is_better,
        load_best_model_at_end=cfg.trainer.load_best_model_at_end,
        optim=cfg.trainer.optim,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        metric_for_best_model=cfg.trainer.metric_for_best_model,
        logging_steps=cfg.trainer.logging_steps,
        warmup_ratio=cfg.trainer.warmup_ratio,
        lr_scheduler_type=cfg.trainer.lr_scheduler_type,
        save_total_limit=cfg.trainer.save_total_limit,
        deepspeed=ds_config,
    )

    # Define callbacks
    callbacks = [
        LogLearningRateCallback(),
        ray.train.huggingface.transformers.RayTrainReportCallback(),
    ]

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # Prepare trainer for Ray
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)

    # Start training
    trainer.train()

    # Save model on the main worker
    if ray.train.get_context().get_world_rank() == 0:
        save_dir = "./saved_model"
        save_model(trainer, save_dir)
        wandb.finish()


@hydra.main(
    version_base=None, config_path="./train_config", config_name="trainer_config"
)
def main(cfg: DictConfig):
    ray.init()

    # Use Ray TorchTrainer instead of HuggingFaceTrainer
    from ray.train.torch import TorchTrainer

    ray_trainer = TorchTrainer(
        train_func,
        train_loop_config=OmegaConf.to_container(cfg, resolve=True),
        scaling_config=ScalingConfig(
            num_workers=cfg.ray.num_workers, 
            resources_per_worker={
                "CPU": 11,
                "GPU": 1,
            },
            use_gpu=cfg.ray.use_gpu
        ),
    )

    # Start training
    result = ray_trainer.fit()

    # Optionally: Load the trained model from checkpoint
    # with result.checkpoint.as_directory() as checkpoint_dir:
    #     checkpoint_path = os.path.join(
    #         checkpoint_dir,
    #         ray.train.huggingface.transformers.RayTrainReportCallback.CHECKPOINT_NAME,
    #     )
    #     model = ESM2AE.from_pretrained(checkpoint_path)

    # Shutdown Ray
    ray.shutdown()

    print("Training finished.")
    print(result)


if __name__ == "__main__":
    main()
