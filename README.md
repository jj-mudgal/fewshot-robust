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

fewshot-robust/..
- attacks.py  
- defenses.py  
- models.py  
- data.py  
- train.py  
- defense_certified.py  
- eval.py  
- utils.py  

results/..  
- plots.py  

scripts/..  
- inspect_dataset.py  

outputs/..  
- fig1_accuracy_comparison.png  
- fig2_trimmed_delta.png  
- fig3_tradeoff.png  

---

## Setup Instructions

### 1. Clone the Repository and Create an Environment

```git clone https://github.com/jj-mudgal/fewshot-robust.git```

```cd fewshot-robust```

```python3 -m venv venv```

```source venv/bin/activate```

### 2. Install Dependencies

```pip install torch torchvision tqdm matplotlib numpy pillow```

### Dataset Setup

The experiments use the Mini-ImageNet dataset, available on Kaggle:
[Mini-ImageNet Dataset (by Arjun Ashok)
](https://www.kaggle.com/datasets/arjunashok33/miniimagenet)

## Verify dataset integrity:

Before training, ensure the Mini-ImageNet dataset is properly organized.
Run the following command to generate a manifest and confirm dataset structure:
python scripts/inspect_dataset.py

This will create a file named dataset_manifest.json in the project root.
It lists the number of classes and images in each split (train, val, test).
Example:
{
"train": {"num_classes": 64},
"val": {"num_classes": 16},
"test": {"num_classes": 20}
}

## Training the Model

5-way 1-shot (Fast, highlights fragility)

```
python train.py --n_way 5 --k_shot 1 --q_queries 15 --episodes 2000 --aq --attack_iters 10
5-way 5-shot (Standard few-shot baseline)


python train.py --n_way 5 --k_shot 5 --q_queries 15 --episodes 2000 --aq --attack_iters 10
10-way 5-shot (Harder configuration)

python train.py --n_way 10 --k_shot 5 --q_queries 15 --episodes 2000 --aq --attack_iters 10
Certified Defense Evaluation
```

Evaluate robustness and trimming-based certified defense:
```
python defense_certified.py --n_way 10 --k_shot 5 --episodes 300 --k_trim 3
```
### Example output:

=== Certified Defense Evaluation ===
Natural accuracy:  19.88%
Trimmed accuracy:  18.81%
Poisoned accuracy: 19.88%
Generating Evaluation Figures

cd results
python plots.py
This will generate:


outputs/..
- fig1_accuracy_comparison.png
- fig2_trimmed_delta.png
- fig3_tradeoff.png

Figure	Description
Fig. 1	Accuracy comparison across few-shot configurations
Fig. 2	Effect of trimming on accuracy (Δ Accuracy)
Fig. 3	Trade-off between natural and trimmed robustness

## Citation
If you use this repository in your research, please cite:

@article{mudgal2025fewshotrobust,
  title     = {The Fragility of Generalization: A Systematic Analysis of Adversarial Attacks and Defenses in Low-Data Learning Regimes},
  author    = {Mudgal, Janmejai and Gupta, Nirmit},
  year      = {2025},
  journal   = {Preprint},
  institution = {Manipal University Jaipur, Department of Computer Science and Engineering},
  note      = {Available at: https://github.com/jj-mudgal/fewshot-robust}
}

License
This repository is licensed under the MIT License.
See the LICENSE file for details.

Contact
For correspondence or research collaboration, contact:
- Janmejai Mudgal — Department of CSE, Manipal University Jaipur (janmejai345@gmail.com)
- Nirmit Gupta — Department of CSE, Manipal University Jaipur (nirmitgupta777@gmail.com)
