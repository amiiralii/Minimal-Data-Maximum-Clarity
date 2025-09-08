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

> *To explain complex tasks, use less (but more informative) data.*

EZR achieves near state-of-the-art optimization performance with a fraction of the labeling cost while producing **transparent, actionable explanations** that surpass attribution-based XAI methods such as LIME, SHAP, and BreakDown.

---

## ✨ Key Contributions

* **Maximum Clarity Heuristic**: Better optimization with fewer but more informative examples.
* **EZR Framework**: Modular pipeline combining active learning, decision trees, and explanation.
* **Efficiency**: Achieves ≥90% of the best-known optimization results in **73% of 60 datasets**, using far fewer labels.
* **Explanations that matter**: Provides global and local cohort-based rationales that are clearer and more actionable than standard XAI methods.
* **Downstream Utility**: Explanation-driven feature selection improves predictive performance.

---

## ⚙️ Repository Structure

```bash
Minimal-Data-Maximum-Clarity/
├── src/                # Source codes for EZR framework
├── datasets/           # 60 datasets from MOOT repository (configs, processes, HPO, etc.)
├── scripts/            # Scripts and configs for replicating results
├── results/            # Experimental results and analysis
├── paper_materials/    # Figures, tables, and supplementary material for the paper
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

## 🔎 Usage

### Run EZR on a Dataset

```bash
python src/run_ezr.py --dataset datasets/coc1000.csv --budget 60
```

* `--dataset`: Path to dataset (tabular format with x/y columns).
* `--budget`: Labeling budget (default: 60).

### Compare Against Baselines

```bash
python experiments/run_baselines.py --dataset datasets/coc1000.csv
```

This benchmarks EZR against regression methods (LR, RF, SVR, ANN, LGBM) and random selection.

---

## 📊 Reproducing Results

The experiments described in the paper can be reproduced with:

```bash
bash experiments/run_all.sh
```

This will:

* Optimize configurations across 60 datasets.
* Generate global & local explanations.
* Compare feature selection methods (EZR, SHAP, ReliefF, ANOVA).
* Output results into the `results/` directory.

---

## 📂 Datasets

We use **60 datasets** from the [MOOT repository](https://github.com/timm/moot/tree/master/optimize), covering:

* **Configurations** (e.g., Apache, SQL, X264)
* **Feature Models** (FFM, FM)
* **Software Process Models** (COC1000, POM3, XOMO)
* **Scrum feature configurations**
* **Miscellaneous datasets** (Wine, auto93)

---

## 📈 Results at a Glance

* **Optimization**: EZR achieves ≥90% of best-known performance in **44/60 datasets (73%)** with minimal labels.
* **Explanations**: Cohort-based explanations outperform LIME, SHAP, and BreakDown in clarity and actionability.
* **Downstream Validation**: Feature rankings from EZR improve regression models on par with or better than standard feature selection methods.

---

## 🧩 Citation

If you use EZR or this repository in your research, please cite our paper:

```bibtex
@article{rayegan2025minimal,
  title={Minimal Data, Maximum Clarity: A Heuristic for Explaining Optimization},
  author={Rayegan, Amirali and Menzies, Tim},
  journal={Journal of Systems and Software},
  year={2025}
}
```

---

## 📜 License

This repository is released under the **MIT License**.

---

## 📬 Contact

For questions or collaboration:

* **Amirali Rayegan** – [GitHub](https://github.com/amiiralii)
* **Tim Menzies** – North Carolina State University

---

👉 Would you like me to also **add badges (build/test status, license, DOI, etc.)** and a **graphical summary figure** from your paper into the README to make it even more journal/professional submission ready?
