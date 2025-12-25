# Neural Network Charity Analysis

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Visualization-purple)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Modeling-yellow)

## Quick Snapshot
I built and evaluated a binary classification model using a deep learning neural network to predict whether a charitable organization’s funding outcome is likely to be **successful** based on historical application data. The final model reached ~72% accuracy, and I documented what limited performance improvements during optimization attempts.

## Table of Contents
- Repo Structure
- Objective
- Dataset Overview
- Build Journey
  - Phase 1 – Data Preprocessing
  - Phase 2 – Baseline Neural Network
  - Phase 3 – Optimization Attempts
- Model Scorecard
- How to Run Locally
- Evidence Gallery
- Key Takeaways
- Next Steps

---

## Repo Structure
Suggested layout (aligns with how I document my other portfolio repos):

- Challenge/
  - (course deliverables / required notebook assets)
- notebooks/
  - 01_preprocessing.ipynb
  - 02_baseline_model.ipynb
  - 03_optimized_model.ipynb
- data/
  - charity_data.csv
- models/
  - AlphabetSoupCharity.h5
  - AlphabetSoupCharity_Optimized.h5
- reports/
  - metrics_baseline.json
  - metrics_optimized.json
  - confusion_matrix.png
  - roc_curve.png
- requirements.txt
- README.md

---

## Objective
Build a model that predicts whether a charity funding application will be **successful** (`IS_SUCCESSFUL`) using historical structured data, then attempt to improve performance through neural network tuning and training strategies.

---

## Dataset Overview
- Target:
  - `IS_SUCCESSFUL` (binary outcome)
- Features:
  - All remaining columns after removing non-predictive identifiers (example: organization IDs / names)
- Notes:
  - Categorical features require encoding (one-hot)
  - Numeric features benefit from scaling
  - Some categories can be too sparse and may need bucketing to reduce noise

---

## Build Journey

<details>
<summary><strong>Phase 1 – Data Preprocessing</strong></summary>

### Objective
Prepare the dataset for model training by defining the target, selecting usable features, and transforming the data into a numeric format suitable for a neural network.

### Actions Taken
- Identified the target variable: `IS_SUCCESSFUL`
- Removed identifier fields that do not help prediction (example: IDs / names)
- Encoded categorical columns (one-hot encoding)
- Scaled numeric columns where appropriate
- Split data into training and testing sets

### Output
A clean training matrix ready for TensorFlow modeling.

</details>

<details>
<summary><strong>Phase 2 – Baseline Neural Network</strong></summary>

### Objective
Create an initial deep learning model and establish a baseline performance level.

### Model Design
- Hidden Layer 1: 80 neurons (ReLU)
- Hidden Layer 2: 30 neurons (ReLU)
- Output Layer: 1 neuron (Sigmoid)

### Result
- Test accuracy: ~72%
- Target goal (75%) was not reached

</details>

<details>
<summary><strong>Phase 3 – Optimization Attempts</strong></summary>

### Objective
Improve model generalization and performance through tuning and training adjustments.

### Actions Taken
- Adjusted number of layers and neurons
- Tested different activation functions
- Applied early stopping during training
- Compared results against traditional ML models (e.g., Random Forest / Logistic Regression)

### Outcome
Performance improvements were limited and did not exceed the 75% target threshold.

</details>

---

## Model Scorecard
| Model Version | Key Changes | Result |
|---|---|---|
| Baseline NN | 2 hidden layers (80 / 30), ReLU + Sigmoid | ~72% accuracy |
| Optimized NN | Architecture + training tweaks (early stopping, etc.) | Did not exceed 75% |

Note: I’m intentionally documenting what *didn’t* work as well as what did — that’s part of doing real technical work.

---

## How to Run Locally

### Option A: Run via Jupyter Notebook
1. Clone the repo:
   git clone https://github.com/JGarza4903/Neural_Network_Charity_Analysis.git
2. Create and activate a virtual environment:
   python -m venv .venv
   .venv\Scripts\activate
3. Install dependencies:
   pip install -r requirements.txt
4. Launch Jupyter:
   jupyter lab
5. Open the notebooks in order:
   - 01_preprocessing.ipynb
   - 02_baseline_model.ipynb
   - 03_optimized_model.ipynb

### Option B: Load the saved model
If you keep the `.h5` model file in `models/`, you can load it in Python:
- Use TensorFlow’s `tf.keras.models.load_model()` and run predictions against preprocessed input.

---

## Evidence Gallery
Placeholders for recruiter-friendly proof (add these as you capture them):
- reports/confusion_matrix.png
- reports/roc_curve.png
- Training history plot (loss/accuracy curves)
- Notebook screenshots of final evaluation output

---

## Key Takeaways
- Neural networks can model complex nonlinear relationships, but performance is heavily influenced by:
  - preprocessing quality
  - category sparsity from one-hot encoding
  - overfitting risk
  - hyperparameter choices
- A “near target” accuracy isn’t a failure — the value is in showing:
  - clean preprocessing
  - repeatable experimentation
  - honest evaluation and iteration

---

## Next Steps
If I revisit this project, I will:
- Add a true baseline scorecard (LogReg first, then NN)
- Improve feature engineering (bucketing rare categories more deliberately)
- Test regularization (dropout, L2) and learning-rate scheduling
- Compare additional classifiers (XGBoost / Gradient Boosting)
- Track experiments in a simple run log (hyperparams → metrics)

---

## About
This repo is part of my portfolio, showing hands-on ML work from preprocessing through model tuning, with an emphasis on reproducible experimentation and honest evaluation.
