# defenses.py
"""
Defense utilities for prototype-based Few-Shot Learning (FSL) models.

This module implements certified and detection-based defense techniques
for episodic few-shot learning, including:
    - Low-level smoothing filters for support regularization
    - Self-similarity scoring for poisoning detection
    - FCert-style trimming defense for certified robustness

These methods can be used to evaluate the reliability and adversarial
resilience of meta-learners such as ProtoNet on benchmarks like Mini-ImageNet.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms


# ============================================================
# Default Support Set Filter
# ============================================================

# Mild Gaussian smoothing for support-set consistency analysis.
_default_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0))


def apply_filter_batch(tensor_batch, filter_fn=None):
    """
    Applies a PIL-based image filter (e.g., Gaussian blur) to a batch of tensors.

    Commonly used for preprocessing or consistency-based defenses that rely on
    low-pass filtering of support images to detect or mitigate adversarial noise.

    Args:
        tensor_batch (Tensor): Batch of images [B, C, H, W] in [0, 1].
        filter_fn (callable, optional): PIL image filter to apply.
            Defaults to a small GaussianBlur kernel.

    Returns:
        Tensor: Filtered image batch on the same device.
    """
    if filter_fn is None:
        filter_fn = _default_blur

    imgs = []
    device = tensor_batch.device

    for i in range(tensor_batch.size(0)):
        img = tensor_batch[i].detach().cpu()
        pil = transforms.ToPILImage()(img)
        pil_f = filter_fn(pil)
        imgs.append(transforms.ToTensor()(pil_f))

    return torch.stack(imgs).to(device)


# ============================================================
# Self-Similarity Defense (Detection Metric)
# ============================================================

def self_similarity_score(model,
                          support,
                          support_y,
                          split_ratio=0.5,
                          filter_fn=None,
                          device=None):
    """
    Computes a self-similarity anomaly score for the support set.

    This metric is *attack-agnostic* and measures the internal consistency
    of a few-shot support set by observing how sensitive predictions are
    to mild input perturbations (e.g., Gaussian blur).

    Concept:
        - Split the support set into auxiliary support (S_aux) and queries (Q_aux).
        - Compute predictions on Q_aux using S_aux as support.
        - Apply a mild filter to S_aux and recompute predictions.
        - The proportion of changed predictions serves as the anomaly score.

    Args:
        model: Few-shot model (e.g., ProtoNet) where model(support, support_y, query) → logits.
        support (Tensor): Support set images [M, C, H, W].
        support_y (Tensor): Support set labels [M].
        split_ratio (float): Fraction of support samples used as S_aux.
        filter_fn (callable, optional): Image filter to apply (default: GaussianBlur).
        device (torch.device, optional): Target device (default: auto-detect).

    Returns:
        float: Self-similarity score ∈ [0, 1], where higher indicates greater anomaly.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    support = support.to(device)
    support_y = support_y.to(device)

    M = support.size(0)
    if M < 2:
        return 0.0

    # Randomly split support into S_aux and Q_aux subsets
    perm = torch.randperm(M)
    split = max(1, int(M * split_ratio))
    s_idx, q_idx = perm[:split], perm[split:]

    # Ensure non-empty query subset
    if q_idx.numel() == 0:
        q_idx = perm[-1:]
        s_idx = perm[:-1] if perm.size(0) > 1 else perm

    S_aux, S_aux_y = support[s_idx], support_y[s_idx]
    Q_aux, Q_aux_y = support[q_idx], support_y[q_idx]

    model.eval()
    with torch.no_grad():
        logits_raw = model(S_aux, S_aux_y, Q_aux)
        preds_raw = logits_raw.argmax(dim=1)

    # Apply smoothing to support subset
    S_filtered = apply_filter_batch(S_aux, filter_fn)

    with torch.no_grad():
        logits_filt = model(S_filtered, S_aux_y, Q_aux)
        preds_filt = logits_filt.argmax(dim=1)

    # Score = fraction of changed predictions
    changed = (preds_raw != preds_filt).float().mean().item()
    return float(changed)


# ============================================================
# Certified Trimming Defense (FCert)
# ============================================================

def trim_and_classify(backbone,
                      support,
                      support_y,
                      query,
                      trim_k=1,
                      device=None):
    """
    Implements the FCert-style trimming defense for prototype-based classifiers.

    The method prunes the farthest support samples (by feature-space distance)
    from each class centroid before computing the final class prototypes.
    This provides robustness guarantees against support-set poisoning.

    Steps:
        1. Extract feature embeddings for support and query sets.
        2. Compute class-wise centroids in the feature space.
        3. For each class, drop the `trim_k` farthest supports.
        4. Recompute trimmed class prototypes.
        5. Classify queries by nearest-prototype distance.

    Args:
        backbone: Feature extractor network f(x) → [B, D].
        support (Tensor): Support set [M, C, H, W].
        support_y (Tensor): Support labels [M].
        query (Tensor): Query set [Q, C, H, W].
        trim_k (int): Number of support samples to drop per class.
        device (torch.device, optional): Computation device (auto-detected if None).

    Returns:
        Tensor: Query logits [Q, N] after classification against trimmed prototypes.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    support = support.to(device)
    support_y = support_y.to(device)
    query = query.to(device)

    with torch.no_grad():
        z_s = backbone(support)   # (M, D)
        z_q = backbone(query)     # (Q, D)

    classes = torch.unique(support_y)
    prototypes = []

    for c in classes:
        idxs = (support_y == c).nonzero(as_tuple=True)[0]
        feats = z_s[idxs]  # (m_c, D)

        centroid = feats.mean(dim=0, keepdim=True)  # (1, D)
        dists = torch.norm(feats - centroid, dim=1)  # (m_c,)

        # Avoid removing all supports for small classes
        trim_k_eff = min(trim_k, feats.size(0) - 1) if feats.size(0) > 1 else 0

        if trim_k_eff > 0:
            _, order = torch.sort(dists, descending=True)
            keep_mask = torch.ones_like(dists, dtype=torch.bool)
            keep_mask[order[:trim_k_eff]] = False
            kept_feats = feats[keep_mask]
        else:
            kept_feats = feats

        proto = kept_feats.mean(dim=0)
        prototypes.append(proto)

    prototypes = torch.stack(prototypes)  # (N, D)

    # Compute squared Euclidean distance between query and class prototypes
    diffs = z_q.unsqueeze(1) - prototypes.unsqueeze(0)
    dists = torch.sum(diffs ** 2, dim=2)
    logits = -dists  # Higher logit = closer prototype

    return logits
