# scripts/inspect_dataset.py
import json
import os

root = "datasets/mini-imagenet"
splits = ["train", "val", "test"]
manifest = {}

for s in splits:
    split_dir = os.path.join(root, s)
    if not os.path.isdir(split_dir):
        continue
    classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    class_info = {}
    for c in classes:
        cpath = os.path.join(split_dir, c)
        imgs = [f for f in os.listdir(cpath) if os.path.isfile(os.path.join(cpath, f))]
        class_info[c] = len(imgs)
    manifest[s] = {"num_classes": len(classes), "classes": class_info}

with open("dataset_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("Wrote dataset_manifest.json")
