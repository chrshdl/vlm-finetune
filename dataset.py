import json
import os
import random

from faker import Faker
from pdf2image import convert_from_path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from document import BillOfLading

schema_json = json.dumps(BillOfLading.model_json_schema(), indent=2)


def export_mlx_dataset(
    pdf_paths: list[str],
    ground_truths: list[dict],
    output_jsonl: str,
    image_out_dir: str,
):
    """
    Converts PDFs to images, then chunks them into higher-resolution slices
    and writes an mlx-vlm compatible JSONL file.
    """
    os.makedirs(image_out_dir, exist_ok=True)

    system_prompt = (
        "You are an expert logistics data extraction AI."
        "Extract the information from the provided document image chunks into a strict JSON "
        f"object that matches the following schema:\n{schema_json}"
    )

    with open(output_jsonl, "w") as f:
        for pdf_path, truth_dict in zip(pdf_paths, ground_truths):
            # convert all pages of the PDF to a list of PIL Images
            images = convert_from_path(pdf_path, dpi=72)
            page_images = [img.convert("RGB") for img in images]

            image_paths = []
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]

            for page_idx, img in enumerate(page_images):
                width, height = img.size

                # split the page into 2 horizontal chunks top and bottom
                num_chunks = 2
                chunk_height = height // num_chunks
                overlap = 50  # we add a 50px overlap so the text on the cut-line isn't destroyed

                for chunk_idx in range(num_chunks):
                    # then we calculate the bounding box (left, top, right, bottom)
                    top = max(0, chunk_idx * chunk_height - overlap)
                    bottom = min(height, (chunk_idx + 1) * chunk_height + overlap)

                    chunk_img = img.crop((0, top, width, bottom))

                    chunk_path = os.path.join(
                        image_out_dir,
                        f"{base_name}_page{page_idx}_chunk{chunk_idx}.jpg",
                    )
                    chunk_img.save(chunk_path)
                    image_paths.append(chunk_path)

            # create one image dict per generated chunk
            user_content = [{"type": "image"} for _ in image_paths]
            user_content.append(
                {"type": "text", "text": "Extract the bill of lading details."}
            )

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {"role": "user", "content": user_content},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": json.dumps(truth_dict)}],
                },
            ]

            record = {"messages": messages, "images": image_paths}
            f.write(json.dumps(record) + "\n")


def generate_synthetic_dataset(index, output_dir="synthetic_dataset"):
    """Generates a single synthetic PDF and its ground truth JSON."""
    os.makedirs(output_dir, exist_ok=True)

    fake = Faker()

    ground_truth = {
        "shipper_name": fake.company(),
        "consignee_name": fake.company(),
        "vessel": f"{fake.word().capitalize()} {random.randint(100, 999)}V",
        "port_of_loading": fake.city(),
        "port_of_discharge": fake.city(),
        "container_numbers": [
            fake.bothify(text="????#######").upper()
            for _ in range(random.randint(1, 4))
        ],
        "total_gross_weight_kg": round(random.uniform(5000.0, 25000.0), 2),
    }

    pdf_path = os.path.join(output_dir, f"bol_{index}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # we add a slight layout variance so the model doesn't overfit to exact pixels
    x_offset = random.randint(-20, 20)
    y_offset = random.randint(-20, 20)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(200 + x_offset, 750 + y_offset, "BILL OF LADING")

    c.setFont("Helvetica", 12)
    c.drawString(
        50 + x_offset, 700 + y_offset, f"Shipper: {ground_truth['shipper_name']}"
    )
    c.drawString(
        50 + x_offset, 670 + y_offset, f"Consignee: {ground_truth['consignee_name']}"
    )
    c.drawString(
        50 + x_offset, 640 + y_offset, f"Vessel/Voyage: {ground_truth['vessel']}"
    )
    c.drawString(
        50 + x_offset,
        610 + y_offset,
        f"Port of Loading: {ground_truth['port_of_loading']}",
    )
    c.drawString(
        50 + x_offset,
        580 + y_offset,
        f"Port of Discharge: {ground_truth['port_of_discharge']}",
    )

    c.drawString(50 + x_offset, 530 + y_offset, "Containers:")
    y_pos = 510 + y_offset
    for container in ground_truth["container_numbers"]:
        c.drawString(70 + x_offset, y_pos, f"- {container}")
        y_pos -= 20

    c.drawString(
        50 + x_offset,
        y_pos - 20,
        f"Total Gross Weight: {ground_truth['total_gross_weight_kg']} KG",
    )

    # 0.5 chance to generate a second page
    if random.choice([True, False]):
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50 + x_offset, 750 + y_offset, "Page 2: Terms and Conditions")
        c.setFont("Helvetica", 10)
        c.drawString(
            50 + x_offset,
            720 + y_offset,
            "1. The carrier shall not be liable for loss or damage resulting from acts of God.",
        )
        c.drawString(
            50 + x_offset,
            700 + y_offset,
            "2. Freight charges are non-refundable once the vessel has departed.",
        )
        c.drawString(
            50 + x_offset,
            680 + y_offset,
            "3. Demurrage charges may apply if containers are not returned within the free time.",
        )

    c.save()
    return pdf_path, ground_truth


def load_or_generate_synthetic_data(data_dir: str, num_docs: int) -> list[tuple[str, dict]]:
    print("Checking for existing synthetic documents...")
    metadata_file = os.path.join(data_dir, "metadata.json")
    os.makedirs(data_dir, exist_ok=True)

    pdf_paths = []
    ground_truths = []

    if os.path.exists(metadata_file):
        print(f"Found existing data in '{data_dir}'. Loading...")
        try:
            with open(metadata_file, "r") as f:
                saved_data = json.load(f)
                pdf_paths = saved_data["pdf_paths"]
                ground_truths = saved_data["ground_truths"]
        except json.JSONDecodeError:
            print("Corrupted metadata found! Forcing regeneration...")
            pdf_paths = []

    if not pdf_paths:
        print(f"Generating {num_docs} synthetic documents...")
        for i in range(num_docs):
            pdf_path, truth = generate_synthetic_dataset(
                i, output_dir=data_dir
            )
            pdf_paths.append(pdf_path)
            ground_truths.append(truth)

        with open(metadata_file, "w") as f:
            json.dump(
                {"pdf_paths": pdf_paths, "ground_truths": ground_truths}, f, indent=2
            )

    return list(zip(pdf_paths, ground_truths))


def split_dataset(combined_data: list, split_ratio: float, seed: int) -> tuple[list, list]:
    random.seed(seed)
    combined = list(combined_data)
    random.shuffle(combined)

    split_idx = int(len(combined) * split_ratio)
    train_data = combined[:split_idx]
    eval_data = combined[split_idx:]
    return train_data, eval_data


def export_mlx_datasets(train_data: list, eval_data: list, out_dir: str):
    train_pdf_paths, train_truths = zip(*train_data) if train_data else ([], [])
    eval_pdf_paths, eval_truths = zip(*eval_data) if eval_data else ([], [])

    os.makedirs(out_dir, exist_ok=True)
    image_out_dir = os.path.join(out_dir, "images")

    print("Exporting train set to MLX format...")
    export_mlx_dataset(
        list(train_pdf_paths),
        list(train_truths),
        os.path.join(out_dir, "train.jsonl"),
        image_out_dir,
    )

    print("Exporting eval set to MLX format...")
    export_mlx_dataset(
        list(eval_pdf_paths),
        list(eval_truths),
        os.path.join(out_dir, "valid.jsonl"),
        image_out_dir,
    )


def prepare_datasets(
    synthetic_data_dir: str,
    num_docs: int,
    split_ratio: float,
    random_seed: int,
    mlx_data_dir: str,
):
    """Handles synthetic data generation, splitting and MLX export."""
    combined = load_or_generate_synthetic_data(synthetic_data_dir, num_docs)
    train_data, eval_data = split_dataset(combined, split_ratio, random_seed)
    export_mlx_datasets(train_data, eval_data, mlx_data_dir)

    print(f"Training set size: {len(train_data)}")
    print(f"Evaluation set size: {len(eval_data)}")
