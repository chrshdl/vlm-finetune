# Qwen2-VL Structured Data Extraction with MLX

This repository demonstrates how to fine-tune `Qwen2-VL-2B` on Apple Silicon to extract strict, structured output from PDF documents.


## Installation

The codebase uses `uv` for managing the dependencies and the virtual environment. Follow the [installation guide for uv](https://docs.astral.sh/uv/#installation).

Now from the repo root execute:

```bash
uv venv
source .venv/bin/activate
uv sync
```


## Apply mlx_vlm hotfix

Currently, the `mlx_vlm library` version`0.3.12` has a tensor shape bug within the `Qwen2-VL` vision encoder. The attention heads in the vision model are not correctly concatenated and transposed before the final output projection. This causes the model to process corrupted visual features.

To successfully train and run inference you must manually patch your local `mlx_vlm` installation.

Find the Qwen2-VL vision module inside your local Python environment:
`.../site-packages/mlx_vlm/models/qwen2_vl/vision.py`

Right after the `attn_outputs` are collected and before the final output projection add these two lines to correctly stitch the heads back together and format the sequence:

```python
# FIX: concatenate attention heads and transpose to the expected shape
output = mx.concatenate(attn_outputs, axis=2)
output = output.transpose(0, 2, 1, 3)
```


# Training

To generate the synthetic dataset and begin the LoRA fine-tuning process run:

```bash
python train.py
```

The model is highly efficient and can learn to generate structured JSON output after 5 epochs.


# Inference

To test the fine-tuned adapters on a newly generated, unseen document run:

```bash
python inference.py


Loading Base Model (Qwen/Qwen2-VL-2B-Instruct) with Adapters (./vlm_checkpoints) using MLX...
#trainable params: 9.232384 M || all params: 1543.714304 M || trainable%: 0.598%
Created unseen document at: unseen_docs/bol_999.pdf
Ground Truth: {
  "shipper_name": "Smith-Vargas",
  "consignee_name": "Zuniga-Vasquez",
  "vessel": "Teach 724V",
  "port_of_loading": "Ballfort",
  "port_of_discharge": "Lake Benjamin",
  "container_numbers": [
    "SFQK2293602",
    "ISXY5874524",
    "OHYV2936741"
  ],
  "total_gross_weight_kg": 9028.33
}

=== Extracting data from unseen_docs/bol_999.pdf ===
Generating structured output...

=== EXTRACTED DATA ===
{
  "shipper_name": "Smith-Vargas",
  "consignee_name": "Zuniga-Vasquez",
  "vessel": "Teach 724V",
  "port_of_loading": "Ballfort",
  "port_of_discharge": "Ballfort",
  "container_numbers": [
    "SFQK2293602",
    "ISXY5874524",
    "SFQK2293602",
    "OHYV2936741"
  ],
  "total_gross_weight_kg": 9028.3
}
```
