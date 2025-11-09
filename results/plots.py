# plots.py
"""
Visualization utilities for evaluation metrics in Few-Shot Robustness Experiments.

This script generates three publication-ready figures:
    Fig. 1 — Accuracy Comparison Across Few-Shot Tasks
    Fig. 2 — Impact of Certified Trimming on Performance
    Fig. 3 — Trade-off Between Natural and Trimmed Robustness

Usage:
    python plots.py

All figures are automatically saved under: `fewshot-robust/outputs/`
"""

import matplotlib.pyplot as plt
import numpy as np
import os


# ============================================================
# Data Setup
# ============================================================

# Experimental results (replace with actual logged results if updated)
tasks = ['20-way 5-shot', '10-way 5-shot', '5-way 10-shot']
natural = [12.28, 19.88, 35.76]
trimmed = [11.47, 18.81, 36.42]
poisoned = [12.28, 19.88, 35.76]

# Create output directory if missing
os.makedirs("../outputs", exist_ok=True)


# ============================================================
# Figure 1 — Accuracy Comparison
# ============================================================

plt.figure(figsize=(6, 4))
plt.plot(tasks, natural, marker='o', label='Natural', linewidth=1.8)
plt.plot(tasks, trimmed, marker='s', label='Trimmed', linewidth=1.8)
plt.plot(tasks, poisoned, marker='^', label='Poisoned', linewidth=1.8)

plt.xlabel('Task Configuration', fontsize=10)
plt.ylabel('Accuracy (%)', fontsize=10)
plt.title('Fig. 1. Accuracy Comparison Across Few-Shot Tasks', fontsize=10, pad=10)
plt.legend(fontsize=9, loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig("../outputs/fig1_accuracy_comparison.png", dpi=300)
plt.close()


# ============================================================
# Figure 2 — Trimming Impact (Δ Accuracy)
# ============================================================

delta = np.array(trimmed) - np.array(natural)

plt.figure(figsize=(6, 4))
bars = plt.bar(tasks, delta, color='gray')

plt.xlabel('Task Configuration', fontsize=10)
plt.ylabel('Δ (Trimmed − Natural) Accuracy (%)', fontsize=10)
plt.title('Fig. 2. Impact of Certified Trimming on Performance', fontsize=10, pad=10)
plt.axhline(0, color='black', linewidth=0.8)

# Annotate bars with Δ values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:+.2f}',
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("../outputs/fig2_trimmed_delta.png", dpi=300)
plt.close()


# ============================================================
# Figure 3 — Robustness vs. Natural Accuracy
# ============================================================

plt.figure(figsize=(6, 4))
plt.scatter(natural, trimmed, color='black', s=60, label='Few-Shot Configurations')

# Annotate each configuration
for i, label in enumerate(tasks):
    plt.text(natural[i] + 0.5, trimmed[i], label, fontsize=9)

# Parity line (no difference between natural and trimmed)
x = np.linspace(10, 40, 100)
plt.plot(x, x, '--', color='gray', linewidth=1, label='Equal Performance Line')

plt.xlabel('Natural Accuracy (%)', fontsize=10)
plt.ylabel('Trimmed Accuracy (%)', fontsize=10)
plt.title('Fig. 3. Trade-off Between Natural and Trimmed Robustness', fontsize=10, pad=10)
plt.legend(fontsize=9, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig("../outputs/fig3_tradeoff.png", dpi=300)
plt.close()


# ============================================================
# Completion Message
# ============================================================

print("✅ All three figures saved successfully in: fewshot-robust/outputs/")
