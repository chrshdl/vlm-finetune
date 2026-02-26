import difflib
import json
import logging
import os

import mlx.optimizers as optim
from datasets import load_dataset
from mlx_vlm import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.trainer import Dataset, Trainer, save_adapter
from mlx_vlm.trainer.utils import (
    apply_lora_layers,
    find_all_linear_names,
    get_peft_model,
)
from mlx_vlm.utils import load, load_image_processor
from tqdm import tqdm


def evaluate_dev(model, processor, valid_file="mlx_dataset/valid.jsonl", num_samples=5):
    print(f"\nEvaluating CER on {num_samples} validation samples...")
    try:
        with open(valid_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return float("inf")

    samples = lines[:num_samples]
    total_cer = 0.0

    for line in samples:
        data = json.loads(line)
        ground_truth = ""
        for msg in data["messages"]:
            if msg["role"] == "assistant":
                ground_truth = msg["content"][0]["text"]
                break

        prompt_messages = [m for m in data["messages"] if m["role"] != "assistant"]
        prompt = processor.apply_chat_template(
            prompt_messages, add_generation_prompt=True
        )
        image_paths = data.get("images", [])

        output = generate(
            model,
            processor,
            prompt,
            image_paths,
            max_tokens=512,
            verbose=False,
            temperature=0.0,
        )
        prediction = output.text.strip()

        similarity = difflib.SequenceMatcher(None, ground_truth, prediction).ratio()
        total_cer += 1.0 - similarity

    return total_cer / len(samples) if samples else float("inf")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def custom_print(*args, **kwargs):
    tqdm.write(" ".join(map(str, args)), **kwargs)


def main(args):
    logger.info(f"\033[32mLoading model from {args.model_path}\033[0m")
    model, processor = load(
        args.model_path, processor_config={"trust_remote_code": True}
    )
    config = model.config.__dict__
    image_processor = load_image_processor(args.model_path)

    logger.info(f"\033[32mLoading dataset from {args.dataset}\033[0m")
    dataset = load_dataset(args.dataset, split=args.split)

    if "messages" not in dataset.column_names:
        raise ValueError("Dataset must have a 'messages' column")
    if "images" not in dataset.column_names:
        raise ValueError("Dataset must have an 'images' column")

    if args.apply_chat_template:
        logger.info("\033[32mApplying chat template to the dataset\033[0m")

        def process_data(examples):
            examples["messages"] = apply_chat_template(
                config=config,
                processor=processor,
                prompt=examples["messages"],
                return_messages=True,
            )
            return examples

        dataset = dataset.map(process_data)

    dataset = Dataset(
        dataset,
        config,
        processor,
        image_processor=image_processor,
        image_resize_shape=args.image_resize_shape,
    )

    adapter_path = args.adapter_path
    if adapter_path:
        logger.info(f"\033[32mResuming from adapter path {adapter_path}\033[0m")
        logger.info(
            "\033[32mLora rank, alpha, and dropout will be loaded from adapter_config.json file\033[0m"
        )

        model = apply_lora_layers(model, adapter_path)

    else:
        logger.info("\033[32mSetting up LoRA\033[0m")

        list_of_modules = find_all_linear_names(model.language_model)
        model = get_peft_model(
            model,
            list_of_modules,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

    logger.info("\033[32mSetting up optimizer\033[0m")
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    logger.info("\033[32mSetting up trainer\033[0m")
    trainer = Trainer(model, optimizer)

    model.train()

    logger.info("\033[32mTraining model\033[0m")

    best_so_far = float("inf")

    for epoch in range(args.epochs):
        if args.steps == 0:
            args.steps = len(dataset) // args.batch_size

        progress_bar = tqdm(range(args.steps), position=0, leave=True)
        for i in progress_bar:
            loss = trainer.train_step(
                dataset[i * args.batch_size : (i + 1) * args.batch_size]
            )
            progress_bar.update(1)
            progress_bar.set_postfix(
                {"Epoch": epoch, "Step": i, "Loss": f"{loss.item():.4f}"}
            )

            if i % args.print_every == 0:
                custom_print(
                    {
                        "Epoch": epoch,
                        "Step": i,
                        "Loss": f"{loss.item():.4f}",
                    }
                )

        valid_file = os.path.join(args.dataset, "valid.jsonl")

        dev_cer = evaluate_dev(model, processor, valid_file=valid_file)
        logger.info(
            f"\033[32m📊 Validation CER after Epoch {epoch}: {dev_cer:.4f}\033[0m"
        )

        if dev_cer < best_so_far:
            best_so_far = dev_cer
            head, tail = os.path.split(args.output_path)
            best_path = os.path.join(head, "best_" + tail) if head else "best_" + tail

            logger.info(
                f"\033[32m🏆 New best model found (CER: {dev_cer:.4f})! Saving to {best_path}\033[0m"
            )
            save_adapter(model, best_path)

        if args.save_after_epoch and (epoch < (args.epochs - 1)):
            head, tail = os.path.split(args.output_path)

            interim_name = f"epoch_{epoch}_{tail}"
            interim_path = os.path.join(head, interim_name) if head else interim_name

            save_adapter(model, interim_path)

    logger.info(f"\033[32mSaving final epoch adapter to {args.output_path}\033[0m")
    save_adapter(model, args.output_path)
