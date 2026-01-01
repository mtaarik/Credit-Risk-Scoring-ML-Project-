# Credit Risk Scoring & Default Prediction

## Project Overview

This repository contains the source code and documentation for the "Credit Risk Scoring" project, conducted as part of the Machine Learning module at ENSIAS (2025-2026).

The primary objective of this project is to develop a robust algorithmic solution to predict the probability of credit card default for the upcoming month. By leveraging historical data from 30,000 clients, we aim to minimize financial losses (False Negatives) while maintaining a viable commercial approval rate.

The approach combines supervised learning for prediction and unsupervised learning for client segmentation, with a specific focus on handling class imbalance and optimizing decision thresholds for banking security.

## Dataset

The analysis is based on the **UCI Default of Credit Card Clients** dataset.
* [cite_start]**Source:** UCI Machine Learning Repository[cite: 5].
* **Volume:** 30,000 observations (Taiwan, April-September 2005).
* **Characteristics:** 24 features (Demographics, Credit History, Bill Statements, Payments).
* **Challenge:** The dataset is highly imbalanced, with only 22% of clients in the default class.

## Methodology

Our pipeline prioritizes financial security ("Recall maximization") over simple accuracy. The methodology is structured as follows:

### 1. Data Engineering
* **Data Cleaning:** Correction of inconsistencies in categorical variables (Education, Marriage).
* **Outlier Management:** Application of **Winsorization** (capping at the 99th percentile) to handle extreme values (VIP clients) without data loss, ensuring stability for distance-based algorithms.
* **Feature Engineering:** Creation of domain-specific variables:
    * `SD_COUNT`: Frequency of severe delays (>= 2 months).
    * `TOTAL_PAY_RATIO`: Solvency indicator.
    * `UTILIZATION_RATIO`: Credit limit utilization rate.

### 2. Supervised Modeling
We implemented and compared multiple algorithms ranging from interpretable models to complex ensemble methods:
* **Naive Bayes:** Used as a baseline with SMOTE and PowerTransformer.
* **Decision Tree:** implemented with balanced class weights for interpretability.
* **Complex Models:** SVM (RBF kernel), KNN, and MLP (Neural Networks).
* **XGBoost (Champion Model):** Optimized using `scale_pos_weight` to handle imbalance natively.
    * **Threshold Tuning:** The decision threshold was shifted from the standard 0.50 to **0.35** based on the Precision-Recall curve analysis.

### 3. Unsupervised Learning
* **Clustering:** K-Means algorithm (k=4, determined by the Elbow method).
* **Validation:** PCA projection confirmed distinct behavioral groups.
* **Insight:** Identification of a specific cluster ("Cluster 2") containing a 60% real default rate, validating the supervised findings without using target labels.

## Repository Structure

The project is organized as follows:

```text
MLPROJECT/
├── data/
│   ├── interim/             # Intermediate transformed data
│   ├── processed/           # Final data for modeling
│   └── raw/                 # Original UCI dataset
├── models/
│   ├── decision_tree_balanced.pkl
│   ├── naive_bayes_optimized.pkl
│   └── xgboost_ultimate.pkl
├── notebooks/
│   ├── 1.0_exploration_EDA.ipynb    # Exploratory Data Analysis
│   ├── 2.0_cleaning_eng.ipynb       # Cleaning & Feature Engineering Pipeline
│   ├── 3.0_NB_AD_XGB.ipynb          # Training: Naive Bayes, Decision Tree, XGBoost
│   ├── 4.0_MLP_KNN_SVM.ipynb        # Training: Neural Nets, KNN, SVM
│   └── NonSupervised.ipynb          # Clustering (K-Means) & PCA
├── venv/                    # Virtual environment
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies