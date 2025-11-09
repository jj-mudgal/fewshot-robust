# eval.py
"""
Few-Shot Learning (FSL) Evaluation Script with Adversarial and Certified Analysis.

Evaluates a trained Prototypical Network under multiple robustness settings:
    - Natural Accuracy (clean evaluation)
    - Robust Accuracy (adversarially perturbed queries via PGD)
    - Detection Score (self-similarity–based support anomaly detection)
    - Optional Support-Set Poisoning Attack (attack success rate)

Compatible with Apple Silicon (torch.mps), CUDA, and CPU.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from models import ConvNetBackbone, ProtoNet
from data import EpisodicDataset
from attacks import pgd_attack_on_queries, poison_support_set
from defenses import self_similarity_score


# ============================================================
# Core Evaluation Logic
# ============================================================

def evaluate(model, loader, device='cuda', pgd=True, poison=False):
    """
    Evaluates a few-shot model over episodic tasks under multiple adversarial settings.

    The evaluation procedure measures:
        1. Natural Accuracy: Baseline accuracy on clean queries.
        2. Robust Accuracy: Accuracy on adversarially perturbed queries (PGD).
        3. Detection Score: Mean self-similarity score of the support set.
        4. Poisoned Accuracy (optional): Accuracy after adversarial poisoning of support set.

    Args:
        model: Trained ProtoNet instance to be evaluated.
        loader (DataLoader): Episodic DataLoader for test episodes.
        device (str or torch.device): Computation device ('cuda', 'mps', or 'cpu').
        pgd (bool): Whether to perform PGD attacks on query samples.
        poison (bool): Whether to perform poisoning attacks on support sets.

    Returns:
        dict: Dictionary containing averaged metrics:
              {
                "natural_acc": float,
                "robust_acc": float,
                "detect_score_mean": float,
                (optional) "poisoned_acc": float,
                (optional) "attack_success_rate": float
              }
    """
    natural_acc_list = []
    robust_acc_list = []
    detect_scores = []
    poison_acc_list = []

    model.eval()

    for support, support_y, query, query_y in tqdm(loader, desc="Evaluating"):
        support = support.squeeze(0).to(device)
        support_y = support_y.squeeze(0).to(device)
        query = query.squeeze(0).to(device)
        query_y = query_y.squeeze(0).to(device)

        # --- (1) Natural Accuracy on clean queries ---
        with torch.no_grad():
            logits = model(support, support_y, query)
        preds = logits.argmax(dim=1)
        natural_acc = (preds == query_y).float().mean().item()
        natural_acc_list.append(natural_acc)

        # --- (2) Robust Accuracy via PGD attack ---
        if pgd:
            adv_query = pgd_attack_on_queries(
                model,
                support, support_y,
                query, query_y,
                eps=8/255, alpha=2/255, iters=10,
                device=device
            )
            with torch.no_grad():
                logits_adv = model(support, support_y, adv_query)
            preds_adv = logits_adv.argmax(dim=1)
            robust_acc = (preds_adv == query_y).float().mean().item()
        else:
            robust_acc = natural_acc
        robust_acc_list.append(robust_acc)

        # --- (3) Self-Similarity Detection Score ---
        score = self_similarity_score(model, support, support_y, device=device)
        detect_scores.append(score)

        # --- (4) Support-Set Poisoning Attack (optional) ---
        if poison:
            poisoned_support = poison_support_set(
                model,
                support, support_y,
                target_queries=query[:1],   # poison against first query sample
                target_q_y=query_y[:1],
                eps=8/255, alpha=2/255, iters=10,
                device=device
            )
            with torch.no_grad():
                logits_pois = model(poisoned_support, support_y, query)
            preds_pois = logits_pois.argmax(dim=1)
            poison_acc = (preds_pois == query_y).float().mean().item()
            poison_acc_list.append(poison_acc)

    # Aggregate results
    results = {
        "natural_acc": float(np.mean(natural_acc_list)),
        "robust_acc": float(np.mean(robust_acc_list)),
        "detect_score_mean": float(np.mean(detect_scores)),
    }

    if poison:
        results["poisoned_acc"] = float(np.mean(poison_acc_list))
        results["attack_success_rate"] = 1.0 - results["poisoned_acc"]

    return results


# ============================================================
# Command-Line Interface
# ============================================================

if __name__ == "__main__":
    import argparse
    from torchvision import datasets, transforms

    parser = argparse.ArgumentParser(
        description="Evaluate Few-Shot Learning robustness under adversarial settings."
    )
    parser.add_argument("--model_path", type=str, default="protonet.pth",
                        help="Path to the trained ProtoNet model checkpoint.")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of few-shot episodes for evaluation.")
    parser.add_argument("--n_way", type=int, default=5, help="Number of classes per episode.")
    parser.add_argument("--k_shot", type=int, default=1, help="Number of support samples per class.")
    parser.add_argument("--q_queries", type=int, default=15, help="Number of query samples per class.")
    parser.add_argument("--no_pgd", action="store_true", help="Disable PGD attack evaluation.")
    parser.add_argument("--poison", action="store_true", help="Enable support-set poisoning evaluation.")
    args = parser.parse_args()

    # Device detection (supports CUDA / MPS / CPU)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load trained ProtoNet model
    backbone = ConvNetBackbone(in_ch=3, out_dim=64).to(device)
    model = ProtoNet(backbone).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Dataset setup (CIFAR-10 used as example; replace with Mini-ImageNet for experiments)
    base = datasets.CIFAR10(root="./datasets", train=False, download=True)
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])

    episodic = EpisodicDataset(
        base,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_queries=args.q_queries,
        episodes=args.episodes,
        image_transform=transform
    )

    loader = DataLoader(episodic, batch_size=1, shuffle=True)

    # Perform evaluation
    results = evaluate(
        model,
        loader,
        device=device,
        pgd=not args.no_pgd,
        poison=args.poison
    )

    # Display results
    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
