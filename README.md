# Minimal Data, Maximum Clarity

**A Heuristic for Explaining Optimization with Less Data**

Authors: *Amirali Rayegan, Tim Menzies*
Affiliation: *North Carolina State University*

---

## 📖 Overview

This repository contains the research materials, code, and datasets supporting the paper:

> **Minimal Data, Maximum Clarity: A Heuristic for Explaining Optimization**
> (submitted to the *Journal of Systems and Software*).

We introduce **EZR**, a lightweight, interpretable, and modular framework for **multi-objective optimization** in software engineering. Unlike traditional optimization methods that require extensive labeled data, EZR embodies the **Maximum Clarity Heuristic**:

> *To explain complex tasks, use less data.*

EZR achieves near state-of-the-art optimization performance with a fraction of the labeling cost while producing **transparent, actionable explanations** that surpass attribution-based XAI methods such as LIME, SHAP, and BreakDown.

---

## ⚙️ Repository Structure

```bash
Minimal-Data-Maximum-Clarity/
├── src/                # Source codes for EZR framework and extensive experiments
├── datasets/           # 60 datasets from MOOT repository
├── scripts/            # Scripts for replicating results
├── results/            # Experimental results and analysis
│   ├── opt_results/    # Per-dataset raw results of RQ1 (optimization)
│   ├── FS_results/     # Per-dataset raw results of RQ3 (feature selection)
│   └── Explanations/   # Explanations outcomes for RQ2
├── paper_materials/    # supplementary materials for the paper
└── README.md           # Project documentation
```

---

## 🚀 Getting Started

### Requirements

* **Python**: >= 3.12

### Required Packages
- `pandas >= 1.3.0`  
- `numpy >= 1.20.0`  
- `scikit-learn >= 0.24.0`  
- `lightgbm >= 3.3.0`  
- `matplotlib >= 3.4.0`  
- `lime >= 0.2.0`  
- `shap >= 0.40.0`  
- `IPython >= 7.0.0`  
- `statsmodels >= 0.13.0`  
- `dalex >= 1.7.0`  
- `torch >= 2.0.0`

### Installation

You can install all dependencies with:

```bash
pip install -r requirements.txt
```

Alternatively, install manually:

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib lime shap IPython statsmodels dalex torch
```

### Custom Modules
The repository includes additional modules required for experiments:
- `ezr` (core optimization engine)  
- `neural_net` (deep learning components)  
- `relieff` (feature selection)  
- `stats` (utility functions)

Ensure these are in your Python path or the same directory as the main scripts.

### Platform Notes
- **Windows**:  
  - Some packages may require *Microsoft Visual C++ Build Tools*.  
  - `lightgbm` may require additional setup.  
  - For PyTorch, follow the [official installation guide](https://pytorch.org/get-started/locally/) to select the correct wheel for CPU/GPU.  
---

## 📊 Reproducing Results

The experiments described in the paper can be reproduced with:

```bash
bash scripts/reproduce.sh
```

This will:

* Run EZR and optimization baselines on all datasets(RQ1).
* Run downstream feature selection methods(RQ3).
* Generate global & local explanations for sample data(coc1000)(RQ2).
* Output raw results and extracted analytics into the `results/` directory.
---

## 📂 Datasets

We use **60 datasets** from the [MOOT repository](https://github.com/timm/moot/tree/master), covering:

* **Configurations**
* **Feature Models**
* **Software Process Models**
* **Scrum feature configurations**
* **Miscellaneous datasets**

---

## 📜 License

This repository is released under the **MIT License**.

---

## 📬 Contact

For questions or collaboration:

* **Amirali Rayegan** – [Website](https://amiiralii.github.io/)
* **Tim Menzies** – [Website](https://timm.fyi/)

---

## 🧩 Citation

If you use EZR or this repository in your research, please cite our paper:

```bibtex
Links to be Updated!
```

