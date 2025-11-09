# data.py
"""
Episodic dataset loader for Few-Shot Learning (FSL) experiments.

This module provides a PyTorch Dataset class that constructs few-shot
episodes from Mini-ImageNet (or similar datasets) for training and
evaluation of meta-learning models. Each episode simulates an N-way
K-shot task structure with a query set.

Compatible with Mini-ImageNet folder structures and 84×84 resolution images.
"""

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ============================================================
# Mini-ImageNet episodic dataset for Few-Shot Learning (FSL)
# ============================================================

class EpisodicDataset(Dataset):
    """
    Constructs episodic few-shot tasks from Mini-ImageNet-style datasets.

    Each __getitem__ call samples an episode consisting of:
        - N classes (n_way)
        - K support samples per class (k_shot)
        - Q query samples per class (n_query)

    The dataset dynamically creates these episodes during iteration,
    enabling flexible training and evaluation without precomputing tasks.

    Args:
        root (str): Root directory containing Mini-ImageNet splits.
        subset (str): Dataset split — one of {"train", "val", "test"}.
        n_way (int): Number of classes per episode.
        k_shot (int): Number of support samples per class.
        n_query (int): Number of query samples per class.
        episodes (int): Number of episodes to generate per epoch.

    Returns (per episode):
        support (Tensor): Support images of shape [N*K, 3, 84, 84].
        support_labels (Tensor): Integer labels for support images.
        query (Tensor): Query images of shape [N*Q, 3, 84, 84].
        query_labels (Tensor): Integer labels for query images.
    """

    def __init__(self, root="mini-imagenet", subset="train",
                 n_way=5, k_shot=1, n_query=15, episodes=1000):
        self.root = os.path.join(root, subset)
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.episodes = episodes

        # Standard preprocessing pipeline for Mini-ImageNet (84×84 resolution)
        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor()
        ])

        # Load class directories and their corresponding image file paths
        self.classes = sorted(os.listdir(self.root))
        self.class_to_images = {
            c: [os.path.join(self.root, c, f)
                for f in os.listdir(os.path.join(self.root, c))]
            for c in self.classes
        }

    def __len__(self):
        """Returns the total number of available episodes."""
        return self.episodes

    def __getitem__(self, idx):
        """
        Samples one few-shot episode consisting of randomly selected classes,
        each containing a support and query subset.

        Args:
            idx (int): Episode index (unused; episodes are sampled randomly).

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                - support: [N*K, 3, 84, 84]
                - support_labels: [N*K]
                - query: [N*Q, 3, 84, 84]
                - query_labels: [N*Q]
        """
        # Randomly select N distinct classes for this episode
        sampled_classes = random.sample(self.classes, self.n_way)

        support_imgs, support_labels = [], []
        query_imgs, query_labels = [], []

        # For each selected class, sample K support and Q query examples
        for label, cls in enumerate(sampled_classes):
            imgs = random.sample(self.class_to_images[cls], self.k_shot + self.n_query)

            support_files = imgs[:self.k_shot]
            query_files = imgs[self.k_shot:]

            # Load and transform support set images
            for img_path in support_files:
                support_imgs.append(self.transform(Image.open(img_path).convert("RGB")))
                support_labels.append(label)

            # Load and transform query set images
            for img_path in query_files:
                query_imgs.append(self.transform(Image.open(img_path).convert("RGB")))
                query_labels.append(label)

        # Convert lists into batched tensors
        support = torch.stack(support_imgs)
        query = torch.stack(query_imgs)
        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)

        return support, support_labels, query, query_labels
