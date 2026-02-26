import os
from dataclasses import dataclass

import custom_trainer
from config import (
    MLX_DATA_DIR,
    MODEL_ID,
    NUM_DOCS,
    OUTPUT_DIR,
    RANDOM_SEED,
    SPLIT_RATIO,
    SYNTHETIC_DATA_DIR,
)
from dataset import prepare_datasets


@dataclass
class MLXArgs:
    model_path: str = MODEL_ID
    dataset: str = MLX_DATA_DIR
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 1e-5
    apply_chat_template: bool = True
    output_path: str = os.path.join(OUTPUT_DIR, "adapters.safetensors")
    save_after_epoch: bool = True
    split: str = "train"
    image_resize_shape: tuple | None = None
    iters: int = 1000
    steps: int = 0
    print_every: int = 10
    lora_rank: int = 8
    lora_alpha: float = 0.1
    lora_dropout: float = 0.1
    adapter_path: str | None = None
    resume_adapter_file: str | None = None


if __name__ == "__main__":
    prepare_datasets(
        synthetic_data_dir=SYNTHETIC_DATA_DIR,
        num_docs=NUM_DOCS,
        split_ratio=SPLIT_RATIO,
        random_seed=RANDOM_SEED,
        mlx_data_dir=MLX_DATA_DIR,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nStarting clean MLX training loop...")

    custom_trainer.main(MLXArgs())
