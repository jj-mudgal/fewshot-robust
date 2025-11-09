# FewShot-Robust: A Systematic Analysis of Adversarial Robustness in Few-Shot Learning

**Authors:**  
Janmejai Mudgal and Nirmit Gupta  
Department of Computer Science and Engineering  
Faculty of Engineering, Manipal University Jaipur, India  

---

## Overview

This repository contains the official implementation for the research paper:

**"The Fragility of Generalization: A Systematic Analysis of Adversarial Attacks and Defenses in Low-Data Learning Regimes"**  
by Janmejai Mudgal and Nirmit Gupta (2025)

Few-shot learning (FSL) aims to train models that generalize effectively from limited examples. However, recent research has revealed that FSL models are disproportionately vulnerable to adversarial perturbations.  
This project provides a systematic analysis of such vulnerabilities under both query and support-set attacks and introduces a certified trimming defense that mitigates these effects.

---

## Features

- Prototypical Network backbone (Conv-4 architecture)
- Adversarial query and support-set poisoning attacks (FGSM and PGD)
- Certified defense with FCert-style trimming
- Evaluation pipeline for natural, robust, and poisoned accuracy
- Visualization utilities for experiment results
- Dataset inspection and manifest generation for reproducibility

---

## Repository Structure

fewshot-robust/
│
├── attacks.py # PGD, FGSM, and support-set poisoning attacks
├── defenses.py # Certified trimming and anomaly detection
├── models.py # ConvNet backbone and ProtoNet implementation
├── data.py # Episodic Mini-ImageNet dataset loader
├── train.py # Prototypical Network training script
├── defense_certified.py # Certified defense evaluation pipeline
├── eval.py # Full evaluation script (natural, robust, poisoned)
├── utils.py # Helper utilities (seeding, tracking, etc.)
├── results/
│ └── plots.py # Plot generation for evaluation figures
├── scripts/
│ └── inspect_dataset.py # Dataset manifest and inspection tool
└── outputs/ # Generated figures and metrics


---

## Setup Instructions

### 1. Clone the Repository and Create an Environment

git clone https://github.com/<your-username>/fewshot-robust.git
cd fewshot-robust
python3 -m venv venv
source venv/bin/activate

### 2. Install Dependencies

pip install torch torchvision tqdm matplotlib numpy pillow
Dataset Setup
The experiments use the Mini-ImageNet dataset, available on Kaggle:
Mini-ImageNet Dataset (by Arjun Ashok)

Expected directory structure:

Copy code
fewshot-robust/
└── datasets/
    └── mini-imagenet/
        ├── train/
        ├── val/
        └── test/
        
## Verify dataset integrity:

python scripts/inspect_dataset.py
Training the Model
5-way 1-shot (Fast, highlights fragility)

python train.py --n_way 5 --k_shot 1 --q_queries 15 --episodes 2000 --aq --attack_iters 10
5-way 5-shot (Standard few-shot baseline)

python train.py --n_way 5 --k_shot 5 --q_queries 15 --episodes 2000 --aq --attack_iters 10
10-way 5-shot (Harder configuration)

python train.py --n_way 10 --k_shot 5 --q_queries 15 --episodes 2000 --aq --attack_iters 10
Certified Defense Evaluation
Evaluate robustness and trimming-based certified defense:


python defense_certified.py --n_way 10 --k_shot 5 --episodes 300 --k_trim 3

### Example output:

=== Certified Defense Evaluation ===
Natural accuracy:  19.88%
Trimmed accuracy:  18.81%
Poisoned accuracy: 19.88%
Generating Evaluation Figures

cd results
python plots.py
This will generate:


outputs/
├── fig1_accuracy_comparison.png
├── fig2_trimmed_delta.png
└── fig3_tradeoff.png
Figure	Description
Fig. 1	Accuracy comparison across few-shot configurations
Fig. 2	Effect of trimming on accuracy (Δ Accuracy)
Fig. 3	Trade-off between natural and trimmed robustness

## Citation
If you use this repository in your research, please cite:

@article{mudgal2025fewshotrobust,
  title={The Fragility of Generalization: A Systematic Analysis of Adversarial Attacks and Defenses in Low-Data Learning Regimes},
  author={Mudgal, Janmejai and Gupta, Nirmit},
  year={2025},
  institution={Manipal University Jaipur},
  note={Preprint}
}
License
This repository is licensed under the MIT License.
See the LICENSE file for details.

Contact
For correspondence or research collaboration, contact:
Janmejai Mudgal — Department of CSE, Manipal University Jaipur (janmejai345@gmail.com)
Nirmit Gupta — Department of CSE, Manipal University Jaipur (nirmitgupta777@gmail.com)
