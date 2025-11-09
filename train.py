# train.py
"""
Training script for Prototypical Networks with optional Adversarial Querying (AQ).

This script trains a few-shot classification model under either standard
or adversarial training regimes. Adversarial Querying (AQ) introduces
PGD-based perturbations on the query set to improve model robustness.

Usage:
    python train.py --episodes 2000 --aq

Recommended dataset structure:
    datasets/
        mini-imagenet/
            train/
            val/
            test/
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms

from data import EpisodicDataset
from models import ConvNetBackbone, ProtoNet
from attacks import pgd_attack_on_queries


# ============================================================
# Training Procedure
# ============================================================

def train(args):
    """
    Trains a Prototypical Network under episodic few-shot learning.

    The training alternates between:
        - Clean (natural) episodes.
        - Optional adversarial episodes with PGD-based query perturbation.

    Args:
        args (argparse.Namespace): Training configuration and hyperparameters.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Using device: {device}]")

    # ------------------------------------------------------------
    # Base Dataset
    # ------------------------------------------------------------
    # CIFAR-10 is used as a lightweight example.
    # Replace with Mini-ImageNet or CIFAR-FS for official few-shot benchmarks.
    base = datasets.CIFAR10(
        root="./datasets",
        train=True,
        download=True
    )

    img_tf = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])

    episodic = EpisodicDataset(
        base_dataset=base,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_queries=args.q_queries,
        episodes=args.episodes,
        image_transform=img_tf
    )

    loader = DataLoader(episodic, batch_size=1, shuffle=True)

    # ------------------------------------------------------------
    # Model Setup
    # ------------------------------------------------------------
    backbone = ConvNetBackbone(in_ch=3, hidden_dim=64, out_dim=64).to(device)
    model = ProtoNet(backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------
    model.train()
    step = 0

    for support, support_y, query, query_y in tqdm(loader, desc="Training Episodes"):
        support = support.squeeze(0).to(device)
        support_y = support_y.squeeze(0).to(device)
        query = query.squeeze(0).to(device)
        query_y = query_y.squeeze(0).to(device)

        # (1) Clean training loss
        logits_clean = model(support, support_y, query)
        loss_clean = F.cross_entropy(logits_clean, query_y)

        # (2) Adversarial Querying (AQ) — PGD attack on query set
        if args.aq:
            adv_query = pgd_attack_on_queries(
                model,
                support, support_y,
                query, query_y,
                eps=args.eps, alpha=args.alpha, iters=args.attack_iters,
                device=device
            )
            logits_adv = model(support, support_y, adv_query)
            loss_adv = F.cross_entropy(logits_adv, query_y)

            # Balanced objective: half clean, half adversarial
            loss = 0.5 * loss_clean + 0.5 * loss_adv
        else:
            loss = loss_clean

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        step += 1
        if step % args.log_every == 0:
            print(f"[Step {step}] Loss: {loss.item():.4f}")

    # ------------------------------------------------------------
    # Save Trained Model
    # ------------------------------------------------------------
    torch.save(model.state_dict(), args.save_path)
    print(f"\n✅ Training complete. Model saved to: {args.save_path}\n")


# ============================================================
# Command-Line Interface
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Prototypical Network for Few-Shot Learning with optional Adversarial Querying (AQ)."
    )

    # Few-shot configuration
    parser.add_argument("--n_way", type=int, default=5, help="Number of classes per episode.")
    parser.add_argument("--k_shot", type=int, default=1, help="Number of support samples per class.")
    parser.add_argument("--q_queries", type=int, default=15, help="Number of query samples per class.")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes.")

    # Optimization parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--save_path", type=str, default="protonet.pth", help="Output path for trained model.")

    # Adversarial Querying (AQ)
    parser.add_argument("--aq", action="store_true", help="Enable Adversarial Querying during training.")
    parser.add_argument("--eps", type=float, default=8/255, help="Perturbation magnitude for AQ (PGD).")
    parser.add_argument("--alpha", type=float, default=2/255, help="Step size for PGD attack.")
    parser.add_argument("--attack_iters", type=int, default=10, help="Number of PGD iterations for AQ.")

    # Logging frequency
    parser.add_argument("--log_every", type=int, default=200, help="Logging interval in training steps.")

    args = parser.parse_args()
    train(args)
