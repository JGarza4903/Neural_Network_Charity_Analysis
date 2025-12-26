# Neural Network Charity Analysis

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-purple)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Modeling-yellow)

## Quick Snapshot
This project applies a deep learning neural network to predict whether a charitable organization’s funding application will be successful based on historical application data. The work demonstrates an end-to-end machine learning workflow, including data preprocessing, neural network design, model evaluation, and documented optimization attempts.

---

## Objective
Build and evaluate a binary classification model that predicts funding success (IS_SUCCESSFUL) using TensorFlow and Keras. Establish a baseline neural network, attempt to improve performance through optimization, and analyze why accuracy gains plateaued.

---

## Dataset Overview

**Target variable**
- IS_SUCCESSFUL — indicates whether an organization used funding effectively.

**Removed columns**
- EIN  
- NAME  

These identifier fields were removed because they do not contribute predictive value.

**Feature categories**
- Application metadata: APPLICATION_TYPE, AFFILIATION, USE_CASE, ORGANIZATION  
- Organizational classification: CLASSIFICATION, STATUS  
- Financial context: INCOME_AMT, ASK_AMT  
- Special conditions: SPECIAL_CONSIDERATIONS  

Categorical features were transformed using one-hot encoding, while numerical values were scaled prior to training the neural network.

---

## Build Journey

### Phase 1 — Data Preprocessing
**Goal:** Prepare raw application data for modeling.

**Actions taken**
- Defined IS_SUCCESSFUL as the prediction target.
- Dropped non-predictive identifier columns.
- Binned rare categorical values to reduce dimensionality and noise.
- Applied one-hot encoding to categorical features.
- Scaled numerical inputs.
- Split the dataset into training and testing subsets.

---

### Phase 2 — Baseline Neural Network
**Model architecture**
- Input layer matched to the encoded feature set
- Hidden layer 1: 80 neurons using ReLU activation
- Hidden layer 2: 30 neurons using ReLU activation
- Output layer: 1 neuron using Sigmoid activation

**Training configuration**
- Loss function: binary cross-entropy  
- Optimizer: Adam  
- Evaluation metric: accuracy  

**Result**
The baseline model achieved approximately 72% accuracy, which did not meet the 75% target threshold.

---

### Phase 3 — Optimization Attempts
**Optimization strategies**
- Adjusted the number of layers and neurons.
- Tested alternative activation functions.
- Binned continuous variables such as ASK_AMT.
- Applied early stopping to reduce overfitting.

**Outcome**
Despite multiple tuning strategies, performance improvements were limited and did not exceed the target accuracy.

---

## Model Scorecard

- Baseline Neural Network  
  Two hidden layers (80 / 30 neurons)  
  Test accuracy: ~72%

- Optimized Variants  
  Architectural and training adjustments  
  Test accuracy: below 75%

---

## How to Run Locally

1. Clone the repository from GitHub.
2. Create and activate a Python virtual environment.
3. Install required dependencies from the requirements file.
4. Launch Jupyter Notebook or Jupyter Lab.
5. Run the notebooks sequentially, starting with preprocessing and ending with optimization.

---

## Evidence Gallery
(Visual evidence to be added as the project evolves.)
- Confusion matrix
- ROC curve
- Training accuracy and loss trends
- Notebook outputs showing final evaluation metrics

---

## Key Takeaways
- Neural networks can struggle with heavily encoded categorical datasets without extensive feature engineering.
- Optimization does not guarantee improved performance, and documenting limitations is part of real-world machine learning.
- Establishing baselines and tracking experimentation decisions strengthens technical credibility.

---

## Next Steps
- Introduce traditional machine learning baselines such as Logistic Regression or Random Forest.
- Explore class imbalance handling techniques.
- Apply regularization strategies including dropout or L2 penalties.
- Track experiments using tools such as TensorBoard or MLflow.

---

## About
This repository is part of my technical portfolio, showcasing applied machine learning work with an emphasis on reproducibility, evaluation, and honest documentation of results.
