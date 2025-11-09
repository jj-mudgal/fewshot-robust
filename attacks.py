# attacks.py
"""
Adversarial and poisoning attacks for Few-Shot Learning (FSL) robustness evaluation.

This module implements both query-space and support-set attacks, designed for
episodic meta-learning frameworks such as ProtoNet. All methods are compatible
with Apple Silicon (torch.mps) and CUDA devices.

Implemented Attacks:
    - FGSM (Fast Gradient Sign Method) on query samples
    - PGD (Projected Gradient Descent) on query samples
    - Support-set poisoning via bilevel surrogate optimization
"""

import torch
import torch.nn.functional as F


def fgsm_attack_on_queries(model, support, support_y, queries, query_y,
                           eps=8/255, device=None):
    """
    Performs a single-step FGSM (Fast Gradient Sign Method) adversarial attack
    on the query set while keeping the support set fixed.

    Args:
        model: Few-shot classifier (e.g., ProtoNet) to be attacked.
        support (Tensor): Support images of shape [N*K, C, H, W].
        support_y (Tensor): Support labels of shape [N*K].
        queries (Tensor): Query images to be perturbed.
        query_y (Tensor): True labels corresponding to query samples.
        eps (float): Maximum perturbation (L∞ bound).
        device (str, optional): Target computation device.

    Returns:
        Tensor: Adversarially perturbed query images.
    """
    device = device or ("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")

    support, support_y = support.to(device), support_y.to(device)
    queries, query_y = queries.to(device), query_y.to(device)

    # Enable gradient computation for queries
    queries_adv = queries.clone().detach().requires_grad_(True)

    # Forward pass and compute cross-entropy loss
    logits = model(support, support_y, queries_adv)
    loss = F.cross_entropy(logits, query_y)
    model.zero_grad()
    loss.backward()

    # Apply perturbation in the direction of the gradient sign
    grad_sign = queries_adv.grad.detach().sign()
    queries_adv = torch.clamp(queries_adv + eps * grad_sign, 0, 1).detach()
    return queries_adv


def pgd_attack_on_queries(model, support, support_y, queries, query_y,
                          eps=8/255, alpha=2/255, iters=10,
                          device=None, random_start=True):
    """
    Conducts a multi-step PGD (Projected Gradient Descent) attack on the query set.

    PGD is an iterative variant of FGSM that performs projected optimization
    within an L∞ ball around the input, resulting in stronger adversarial examples.

    Args:
        model: Few-shot classifier under attack.
        support (Tensor): Support set images [N*K, C, H, W].
        support_y (Tensor): Support labels [N*K].
        queries (Tensor): Query images to be attacked.
        query_y (Tensor): True labels for queries.
        eps (float): Maximum perturbation magnitude.
        alpha (float): Step size per iteration.
        iters (int): Number of PGD iterations.
        device (str, optional): Device to execute the attack on.
        random_start (bool): Whether to add initial random noise.

    Returns:
        Tensor: Adversarially perturbed query images.
    """
    device = device or ("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")

    support, support_y = support.to(device), support_y.to(device)
    queries, query_y = queries.to(device), query_y.to(device)

    # Optional random initialization within epsilon-ball
    if random_start:
        queries_adv = queries + torch.empty_like(queries).uniform_(-eps, eps)
    else:
        queries_adv = queries.clone()
    queries_adv = torch.clamp(queries_adv, 0, 1).detach().requires_grad_(True)

    for _ in range(iters):
        logits = model(support, support_y, queries_adv)
        loss = F.cross_entropy(logits, query_y)
        model.zero_grad()
        loss.backward()

        # Gradient ascent step
        grad_sign = queries_adv.grad.detach().sign()
        queries_adv = queries_adv + alpha * grad_sign

        # Project perturbations back into epsilon-ball
        queries_adv = torch.max(torch.min(queries_adv, queries + eps), queries - eps)
        queries_adv = torch.clamp(queries_adv, 0, 1).detach().requires_grad_(True)

    return queries_adv.detach()


def poison_support_set(model, support, support_y,
                       target_queries, target_q_y,
                       eps=8/255, alpha=2/255, iters=10,
                       device=None):
    """
    Performs a support-set poisoning attack via bilevel optimization.

    The attack perturbs support samples to maximize classification error
    on a given set of target query samples. This simulates data poisoning
    scenarios in episodic few-shot learning.

    Args:
        model: Few-shot learner under attack.
        support (Tensor): Support images [N*K, C, H, W].
        support_y (Tensor): Labels for support images [N*K].
        target_queries (Tensor): Query images to degrade performance on.
        target_q_y (Tensor): True labels for target queries.
        eps (float): Maximum allowed perturbation.
        alpha (float): Step size for gradient ascent.
        iters (int): Number of bilevel update steps.
        device (str, optional): Target computation device.

    Returns:
        Tensor: Poisoned (adversarially modified) support set.
    """
    device = device or ("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")

    support_orig = support.to(device)
    support_y = support_y.to(device)
    target_queries = target_queries.to(device)
    target_q_y = target_q_y.to(device)

    # Initialize poisoned copy of support images
    poisoned = support_orig.clone().detach().requires_grad_(True)

    for _ in range(iters):
        logits = model(poisoned, support_y, target_queries)
        loss = F.cross_entropy(logits, target_q_y)
        model.zero_grad()
        loss.backward()

        # Gradient ascent on support samples
        grad_sign = poisoned.grad.detach().sign()
        poisoned = poisoned + alpha * grad_sign

        # Project perturbations into epsilon-ball and valid pixel range
        poisoned = torch.max(torch.min(poisoned, support_orig + eps), support_orig - eps)
        poisoned = torch.clamp(poisoned, 0, 1).detach().requires_grad_(True)

    return poisoned.detach()
