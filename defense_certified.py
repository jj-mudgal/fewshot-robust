# defense_certified.py
"""
Certified Defense Evaluation for Few-Shot Learning (FSL) Models.

This script evaluates few-shot classifiers under natural, trimmed (certified),
and adversarially poisoned conditions. It uses the episodic evaluation protocol
on datasets such as Mini-ImageNet and supports Apple Silicon (torch.mps) and CUDA.

Implements:
    - Certified trimming defense evaluation (Trim-k)
    - Optional support-set poisoning attacks
    - Standardized metrics reporting for reproducibility
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from models import ConvNetBackbone, ProtoNet
from data import EpisodicDataset
from attacks import poison_support_set
from defenses import trim_and_classify


# ============================================================
# Certified Defense Evaluation
# ============================================================

def evaluate_trim_certified(model, backbone, loader, device, k_trim=1, attack_poison_support=False):
    """
    Evaluates certified (Trim-k) robustness of a Few-Shot model.

    This function computes three key metrics:
        1. Natural Accuracy — on clean support/query sets.
        2. Trimmed Accuracy — using certified prototype trimming (Trim-k).
        3. Poisoned Accuracy — after applying support-set poisoning (optional).

    Args:
        model: Few-shot learner (e.g., ProtoNet) under evaluation.
        backbone: Feature extractor network used by the model.
        loader (DataLoader): Episodic evaluation loader.
        device (torch.device): Target computation device.
        k_trim (int): Number of support samples trimmed per class (Trim-k).
        attack_poison_support (bool): Whether to simulate support-set poisoning.

    Returns:
        dict: Averaged evaluation metrics containing:
              - "natural_acc"
              - "trimmed_acc"
              - "poisoned_acc"
    """
    model.eval()
    nat_accs, trim_accs, pois_accs = [], [], []

    for batch in tqdm(loader, total=len(loader)):
        support, support_y, query, query_y = batch
        support = support.squeeze(0).to(device)
        support_y = support_y.squeeze(0).to(device)
        query = query.squeeze(0).to(device)
        query_y = query_y.squeeze(0).to(device)

        # --- 1. Natural accuracy (clean evaluation) ---
        with torch.no_grad():
            logits_nat = model(support, support_y, query)
            nat_acc = (logits_nat.argmax(1) == query_y).float().mean().item()

        # --- 2. Trimmed (certified) accuracy ---
        with torch.no_grad():
            logits_trim = trim_and_classify(backbone, support, support_y, query,
                                            trim_k=k_trim, device=device)
            trim_acc = (logits_trim.argmax(1) == query_y).float().mean().item()

        # --- 3. Poisoned support-set attack (optional) ---
        if attack_poison_support:
            poisoned_support = poison_support_set(model, support, support_y,
                                                  query[:1], query_y[:1],
                                                  eps=8/255, alpha=2/255,
                                                  iters=10, device=device)
            with torch.no_grad():
                logits_pois = model(poisoned_support, support_y, query)
                pois_acc = (logits_pois.argmax(1) == query_y).float().mean().item()
        else:
            pois_acc = nat_acc

        # Record episode-level results
        nat_accs.append(nat_acc)
        trim_accs.append(trim_acc)
        pois_accs.append(pois_acc)

    return {
        "natural_acc": sum(nat_accs) / len(nat_accs),
        "trimmed_acc": sum(trim_accs) / len(trim_accs),
        "poisoned_acc": sum(pois_accs) / len(pois_accs),
    }


# ============================================================
# Main Execution Pipeline
# ============================================================

def main(data_root, split, n_way, k_shot, q_queries, episodes, k_trim, attack_poison_support):
    """
    Main entry point for certified defense evaluation.

    Steps:
        1. Load the specified dataset split (train/val/test).
        2. Instantiate episodic loader for N-way K-shot tasks.
        3. Load pretrained ProtoNet model and backbone.
        4. Evaluate model under natural, trimmed, and (optionally) poisoned settings.

    Args:
        data_root (str): Root path to dataset directory.
        split (str): Dataset split ("train", "val", or "test").
        n_way (int): Number of classes per episode.
        k_shot (int): Support samples per class.
        q_queries (int): Query samples per class.
        episodes (int): Number of episodes for evaluation.
        k_trim (int): Number of samples trimmed in certified defense.
        attack_poison_support (bool): Whether to apply poisoning attacks.
    """
    # --- Device Selection (supports Apple MPS, CUDA, or CPU) ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Dataset Preparation ---
    split_root = os.path.join(data_root, split)
    if not os.path.exists(split_root):
        raise FileNotFoundError(f"Dataset split not found: {split_root}")

    dataset = EpisodicDataset(
        root=data_root, subset=split,
        n_way=n_way, k_shot=k_shot,
        n_query=q_queries, episodes=episodes
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # --- Model Initialization ---
    backbone = ConvNetBackbone(in_ch=3, out_dim=64)
    model = ProtoNet(backbone).to(device)

    if not os.path.exists("protonet.pth"):
        raise FileNotFoundError(
            "Missing model checkpoint: protonet.pth. "
            "Please train the model first using train.py."
        )

    model.load_state_dict(torch.load("protonet.pth", map_location=device))
    model.eval()

    # --- Certified Defense Evaluation ---
    metrics = evaluate_trim_certified(
        model, backbone, loader, device,
        k_trim=k_trim, attack_poison_support=attack_poison_support
    )

    # --- Report Results ---
    print("\n=== Certified Defense Evaluation ===")
    print(f"Natural accuracy:  {metrics['natural_acc'] * 100:.2f}%")
    print(f"Trimmed accuracy:  {metrics['trimmed_acc'] * 100:.2f}%")
    print(f"Poisoned accuracy: {metrics['poisoned_acc'] * 100:.2f}%")


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Certified defense evaluation for Few-Shot Learning models."
    )
    parser.add_argument("--data_root", type=str, default="./datasets/mini-imagenet",
                        help="Root directory of dataset (Mini-ImageNet recommended).")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split: train / val / test.")
    parser.add_argument("--n_way", type=int, default=5, help="Number of classes per episode.")
    parser.add_argument("--k_shot", type=int, default=5, help="Number of support samples per class.")
    parser.add_argument("--q_queries", type=int, default=15, help="Number of query samples per class.")
    parser.add_argument("--episodes", type=int, default=200, help="Number of evaluation episodes.")
    parser.add_argument("--k_trim", type=int, default=1, help="Trim-k parameter for certified defense.")
    parser.add_argument("--attack_poison_support", action="store_true",
                        help="Enable support-set poisoning attacks.")
    args = parser.parse_args()

    main(args.data_root, args.split, args.n_way, args.k_shot,
         args.q_queries, args.episodes, args.k_trim, args.attack_poison_support)
