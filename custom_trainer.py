import json
import logging
import os
import re

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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> str:
    """Extracts the JSON string from markdown."""
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def calculate_dict_accuracy(truth_dict: dict, pred_dict: dict) -> float:
    """
    Calculates the accuracy between two dictionaries on field level.
    Handles lists by comparing them as sets, which are order-invariant.
    """
    if not truth_dict:
        return 0.0

    correct_fields = 0
    total_fields = len(truth_dict)

    for key, truth_val in truth_dict.items():
        pred_val = pred_dict.get(key)

        if isinstance(truth_val, list) and isinstance(pred_val, list):
            if set(truth_val) == set(pred_val):
                correct_fields += 1
        elif truth_val == pred_val:
            correct_fields += 1

    return correct_fields / total_fields


def evaluate_dev(model, processor, valid_file="mlx_dataset/valid.jsonl", num_samples=5):
    print(f"\nEvaluating Field-Level Accuracy on {num_samples} validation samples...")
    try:
        with open(valid_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return 0.0  # if file is missing return accuracy 0

    samples = lines[:num_samples]
    total_accuracy = 0.0

    for line in samples:
        data = json.loads(line)
        ground_truth_full = ""
        for msg in data["messages"]:
            if msg["role"] == "assistant":
                ground_truth_full = msg["content"][0]["text"]
                break

        ground_truth_str = extract_json_from_text(ground_truth_full)
        try:
            ground_truth_dict = json.loads(ground_truth_str)
        except json.JSONDecodeError:
            ground_truth_dict = {}

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
            max_tokens=1024,
            verbose=False,
            temperature=0.0,
        )

        prediction_full = output.text.strip()
        prediction_str = extract_json_from_text(prediction_full)

        try:
            prediction_dict = json.loads(prediction_str)
        except json.JSONDecodeError:
            prediction_dict = {}

        accuracy = calculate_dict_accuracy(ground_truth_dict, prediction_dict)
        total_accuracy += accuracy

    mean_accuracy = total_accuracy / len(samples) if samples else 0.0
    return mean_accuracy


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

    best_so_far = 0.0

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

        dev_accuracy = evaluate_dev(model, processor, valid_file=valid_file)
        logger.info(
            f"\033[32m📊 Validation Accuracy after Epoch {epoch}: {dev_accuracy:.2%}\033[0m"
        )

        if dev_accuracy > best_so_far:
            best_so_far = dev_accuracy
            head, tail = os.path.split(args.output_path)
            best_path = os.path.join(head, "best_" + tail) if head else "best_" + tail

            logger.info(
                f"\033[32m🏆 New best model found (Accuracy: {dev_accuracy:.4f})! Saving to {best_path}\033[0m"
            )
            save_adapter(model, best_path)

        if args.save_after_epoch and (epoch < (args.epochs - 1)):
            head, tail = os.path.split(args.output_path)

            interim_name = f"epoch_{epoch}_{tail}"
            interim_path = os.path.join(head, interim_name) if head else interim_name

            save_adapter(model, interim_path)

    logger.info(f"\033[32mSaving final epoch adapter to {args.output_path}\033[0m")
    save_adapter(model, args.output_path)
