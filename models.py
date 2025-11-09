# models.py
"""
Model architectures for Few-Shot Learning (FSL) and Prototypical Networks.

This module defines:
    - ConvBlock: A standard CNN feature extraction block
    - ConvNetBackbone: The canonical 4-layer few-shot convolutional encoder
    - euclidean_dist: Pairwise squared Euclidean distance computation
    - ProtoNet: Prototypical Network (Snell et al., 2017)

References:
    Snell, J., Swersky, K., & Zemel, R. (2017). 
    "Prototypical Networks for Few-Shot Learning."
    Advances in Neural Information Processing Systems (NeurIPS).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Feature Extraction Backbone
# ============================================================

class ConvBlock(nn.Module):
    """
    A single convolutional block: Conv → BatchNorm → ReLU → MaxPool.

    Widely adopted as the basic building block for few-shot feature extractors.
    It balances model capacity and sample efficiency.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        """
        Forward pass for a single feature block.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Transformed tensor with reduced spatial dimensions.
        """
        return self.layer(x)


class ConvNetBackbone(nn.Module):
    """
    Canonical 4-layer convolutional backbone used in few-shot literature.

    Input:
        Images of shape (B, 3, H, W), typically 84×84 (Mini-ImageNet standard).

    Output:
        Flattened feature vectors suitable for metric-based classification.

    Args:
        in_ch (int): Input channels (default 3 for RGB).
        hidden_dim (int): Channels in intermediate layers (default 64).
        out_dim (int): Output embedding dimensionality (default 64).
    """
    def __init__(self, in_ch=3, hidden_dim=64, out_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_ch, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, out_dim)
        )

    def forward(self, x):
        """
        Encodes input images into compact feature embeddings.

        Args:
            x (Tensor): Image batch of shape (B, C, H, W).

        Returns:
            Tensor: Flattened embeddings (B, D).
        """
        out = self.encoder(x)
        return out.view(out.size(0), -1)


# ============================================================
# Distance Function
# ============================================================

def euclidean_dist(x, y):
    """
    Computes pairwise squared Euclidean distances between two sets of vectors.

    Args:
        x (Tensor): Query feature matrix of shape (n, d).
        y (Tensor): Prototype feature matrix of shape (m, d).

    Returns:
        Tensor: Distance matrix of shape (n, m),
                where entry (i, j) = ||x_i - y_j||².
    """
    n, m, d = x.size(0), y.size(0), x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(dim=2)


# ============================================================
# Prototypical Network
# ============================================================

class ProtoNet(nn.Module):
    """
    Prototypical Network for Few-Shot Learning.

    Given a support set (labeled examples) and a query set (unlabeled examples),
    this model embeds both via a shared backbone, constructs class prototypes,
    and classifies queries by nearest-prototype distance.

    Args:
        backbone (nn.Module): Feature extractor returning embeddings.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, support, support_y, query):
        """
        Performs forward inference for few-shot classification.

        Steps:
            1. Embed support and query examples via backbone.
            2. Compute per-class prototypes (mean embedding per label).
            3. Compute Euclidean distance between each query and prototypes.
            4. Return logits = -distance (higher → closer).

        Args:
            support (Tensor): Support images [N*K, C, H, W].
            support_y (Tensor): Support labels [N*K].
            query (Tensor): Query images [Q, C, H, W].

        Returns:
            Tensor: Logits [Q, N] representing negative distances to prototypes.
        """
        # Feature embeddings
        z_support = self.backbone(support)  # (N*K, D)
        z_query = self.backbone(query)      # (Q, D)

        # Compute class prototypes
        classes = torch.unique(support_y)
        prototypes = [z_support[support_y == c].mean(dim=0) for c in classes]
        prototypes = torch.stack(prototypes)  # (N, D)

        # Compute negative Euclidean distances as logits
        dists = euclidean_dist(z_query, prototypes)
        logits = -dists
        return logits
