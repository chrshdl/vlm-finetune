import json
import os
import shutil
import sys

from mlx_vlm import generate, load
from pdf2image import convert_from_path
from pydantic import ValidationError

from config import MODEL_ID, OUTPUT_DIR
from dataset import generate_synthetic_dataset
from document import BillOfLading

base_checkpoint_dir = OUTPUT_DIR
adapter_path = None

best_dir = os.path.join(base_checkpoint_dir, "best")
best_file = os.path.join(base_checkpoint_dir, "best_adapters.safetensors")

if os.path.exists(best_dir):
    adapter_path = best_dir

elif os.path.exists(best_file):
    print("Found 'best_adapters.safetensors'! Preparing isolated load directory...")

    os.makedirs(best_dir, exist_ok=True)

    # copy config and rename weights to satisfy MLX's naming rules
    shutil.copy(
        os.path.join(base_checkpoint_dir, "adapter_config.json"),
        os.path.join(best_dir, "adapter_config.json"),
    )
    shutil.copy(best_file, os.path.join(best_dir, "adapters.safetensors"))

    adapter_path = best_dir

if not adapter_path:
    print("No best model found... I quit")
    sys.exit(0)


print(f"Loading Base Model ({MODEL_ID}) with Adapters ({adapter_path}) using MLX...")
model, processor = load(MODEL_ID, adapter_path=adapter_path)

schema_json = json.dumps(BillOfLading.model_json_schema(), indent=2)


def extract_logistics_data(pdf_path: str):
    print(f"\n=== Extracting data from {pdf_path} ===")

    # DPI shall match value as in training
    images = convert_from_path(pdf_path, dpi=72)
    image_paths = []

    out_dir = "unseen_docs_images"
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_idx, img in enumerate(images):
        width, height = img.size

        num_chunks = 2
        chunk_height = height // num_chunks
        overlap = 50

        for chunk_idx in range(num_chunks):
            top = max(0, chunk_idx * chunk_height - overlap)
            bottom = min(height, (chunk_idx + 1) * chunk_height + overlap)

            chunk_img = img.crop((0, top, width, bottom))

            p = os.path.join(
                out_dir, f"{base_name}_page{page_idx}_chunk{chunk_idx}.jpg"
            )
            chunk_img.save(p)
            image_paths.append(p)

    user_content = [{"type": "image"} for _ in image_paths]
    user_content.append({"type": "text", "text": "Extract the bill of lading details."})

    system_prompt = (
        "You are an expert logistics data extraction AI. "
        "Extract the information from the provided document image chunks into a strict JSON "
        f"object that matches the following schema:\n{schema_json}"
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    print("Generating structured output...")
    output = generate(
        model,
        processor,
        prompt,
        image_paths,
        max_tokens=512,
        verbose=False,
        temperature=0.0,
    )

    json_output = output.text

    if "```json" in json_output:
        json_output = json_output.split("```json")[-1].split("```")[0].strip()
    elif "```" in json_output:
        json_output = json_output.split("```")[-1].strip()

    try:
        parsed_data = BillOfLading.model_validate_json(json_output)
        return parsed_data
    except ValidationError:
        print("Failed to parse JSON.")
        print("Raw output:", json_output)
        return None


# generate an unseen document
unseen_pdf_path, unseen_ground_truth = generate_synthetic_dataset(
    999, output_dir="unseen_docs"
)

print(f"Created unseen document at: {unseen_pdf_path}")
print("\n=== GROUND TRUTH ===")
print(json.dumps(unseen_ground_truth, indent=2))

extracted_data = extract_logistics_data(unseen_pdf_path)

if extracted_data:
    print("\n=== EXTRACTED DATA ===")
    print(json.dumps(extracted_data.model_dump(), indent=2))
