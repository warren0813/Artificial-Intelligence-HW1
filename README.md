# Artificial-Intelligence-HW1

Homework project for multiclass healthy-food classification using engineered nutrition features.

## Overview

This repository contains two notebook-based tasks for predicting `health_label` (`Healthy`, `Moderate`, `Unhealthy`) from nutrition features in `foods_engineered_300.csv`.

Both tasks follow a similar workflow:
1. Data validation and quick EDA
2. Stratified cross-validation training and evaluation
3. Results analysis (confusion matrix + macro metrics)
4. Required experiments (A-D)

## Tasks

### Task 1: XGBoost Classifier
- Notebook: `classifier_xgboost.ipynb`
- Model: `XGBClassifier` (multiclass softprob)
- Evaluation: accuracy, macro precision/recall/F1, AUROC (OvR macro), confusion matrix

### Task 2: PyTorch MLP Classifier
- Notebook: `classifier_pytorch_mlp.ipynb`
- Model: fully connected neural network (MLP) with dropout and early stopping
- Evaluation: accuracy, macro precision/recall/F1, AUROC (OvR macro), confusion matrix

## Experiments Implemented (Both Tasks)

- A) Training-data amount vs performance
- B) Simulated class imbalance: baseline vs weighting vs SMOTE
- C) Data augmentation (tabular Gaussian noise) vs none
- D) Dimensionality reduction (PCA) vs none

## Dataset

- Primary file: `foods_engineered_300.csv`
- Compatibility fallback used in notebooks: `dataset_preprocessing/foods_engineered_300.csv`

## Environment and Dependencies

Use Python 3.10+ (or a recent Conda environment).

Install required packages:

```bash
pip install xgboost scikit-learn imbalanced-learn pandas numpy matplotlib seaborn torch
```

## How to Run

1. Open this project in VS Code (or Jupyter Lab).
2. Start with either notebook:
   - `classifier_xgboost.ipynb`
   - `classifier_pytorch_mlp.ipynb`
3. Run cells top-to-bottom.
4. Review:
   - CV fold metrics and mean/std summary
   - Confusion matrix
   - Experiment tables (`expA`, `expB`, `expC`, `expD`)

## Expected Outputs for Report

- Baseline cross-validation performance for each model
- Comparison of experiment settings (A-D)
- Discussion of class confusion patterns and macro-F1 behavior
- Final comparison between XGBoost and PyTorch MLP on this dataset

## Goal

Compare model performance and behavior under different data and preprocessing settings for robust `health_label` prediction.
