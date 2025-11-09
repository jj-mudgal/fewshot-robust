# scripts/inspect_dataset.py
"""
Dataset inspection and manifest generation script for Mini-ImageNet or similar datasets.

This utility scans the dataset directory structure and records:
    - Number of classes per split (train/val/test)
    - Number of images per class
    - Directory structure integrity

Outputs a JSON manifest for easy verification and reproducibility tracking.

Usage:
    python scripts/inspect_dataset.py

Expected folder layout:
    datasets/
        mini-imagenet/
            train/
                n01440764/
                    image1.JPEG
                    image2.JPEG
                    ...
            val/
            test/

Result:
    dataset_manifest.json — written in project root.
"""

import json
import os


# ============================================================
# Configuration
# ============================================================

root = "datasets/mini-imagenet"
splits = ["train", "val", "test"]
manifest = {}


# ============================================================
# Dataset Inspection
# ============================================================

for s in splits:
    split_dir = os.path.join(root, s)
    if not os.path.isdir(split_dir):
        print(f"⚠️ Warning: Missing split directory — {split_dir}")
        continue

    # Identify valid class subfolders
    classes = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])

    class_info = {}

    for c in classes:
        cpath = os.path.join(split_dir, c)
        imgs = [
            f for f in os.listdir(cpath)
            if os.path.isfile(os.path.join(cpath, f))
        ]
        class_info[c] = len(imgs)

    manifest[s] = {
        "num_classes": len(classes),
        "classes": class_info
    }

# ============================================================
# Save Manifest
# ============================================================

output_path = "dataset_manifest.json"

with open(output_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"✅ Dataset manifest successfully written to: {output_path}")
