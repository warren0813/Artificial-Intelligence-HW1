# Artificial-Intelligence-HW1

Homework project for multiclass healthy-food classification using engineered nutrition features.

## What It Includes
- Data validation and exploratory analysis
- Two model notebooks:
	- XGBoost classifier
	- PyTorch MLP classifier
- Stratified cross-validation evaluation with accuracy, macro precision/recall/F1, AUROC, and confusion matrices
- Experiments on:
	- training-data size
	- class imbalance handling (class weighting / SMOTE)
	- data augmentation
	- dimensionality reduction (PCA)

## Dataset
- `dataset_preprocessing/foods_engineered_300.csv`

## Goal
Compare model performance and behavior under different data and preprocessing settings for `health_label` prediction.
