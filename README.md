# Machine Learning Assignment 2 | Breast Cancer Classification

## A. Problem statement

Predict whether a breast mass is benign or malignant from 30 numeric features extracted from images of fine needle aspirates. Compare five binary classifiers on the same held-out test set using accuracy, ROC AUC, precision, recall, F1 score, and Matthews correlation coefficient (MCC).

## B. Dataset description

The [UCI Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) contains 569 observations and 30 numeric input features, exceeding the assignment minimum of 500 observations and 12 features. The features summarize characteristics of cell nuclei, such as radius, texture, and perimeter. The dataset has 357 benign and 212 malignant observations.

The training program loads the copy bundled with scikit-learn. It uses the 30 numeric features and encodes the target as **0 = benign** and **1 = malignant**. An 80/20 stratified split with seed 42 produces 455 training rows and 114 test rows. The exported `test_data.csv` contains those 114 test rows with their feature values and target labels.

## C. GitHub repository link

**Repository URL** — https://github.com/ajeeshakd/ml-assignment-2

The repository contains `train_models.py`, `streamlit_app.py`, `requirements.txt`, `README.md`, `test_data.csv`, and `model/` with the five saved pipelines and their metadata.

## D. Models used

The five classifiers are Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN, with `k=5`), Gaussian Naive Bayes, and a Support Vector Machine (SVM) with an RBF kernel. The Logistic Regression, KNN, and SVM pipelines standardize features using statistics learned from training data. All five models are evaluated on the same 114 test rows. Precision, recall, and F1 use malignant as the positive class. ROC AUC uses positive-class probabilities or, for SVM, decision scores.

### Evaluation metrics comparison

> The table reports the held-out metrics from the included `results/model_comparison.csv`.

| ML model name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9649 | **0.9960** | 0.9750 | **0.9286** | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | **1.0000** | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | **0.9737** | 0.9947 | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

### Model performance observations

| ML model name | Observation about model performance |
| --- | --- |
| Logistic Regression | Achieved the highest ROC AUC (0.9960) and detected 39 of the 42 malignant cases. It also had the highest mean F1 (0.9640) in five-fold cross-validation on the training data. |
| Decision Tree | Recorded the lowest test accuracy (0.9298) and MCC (0.8492). It missed four malignant cases and incorrectly flagged four benign cases. |
| KNN | Reached 0.9561 accuracy and 0.9383 F1. It detected 38 of the 42 malignant cases and incorrectly flagged one benign case. |
| Gaussian Naive Bayes | Correctly classified every case it predicted as malignant, giving 1.0000 precision. It missed seven malignant cases, giving it the lowest recall (0.8333). |
| Support Vector Machine | Led on test accuracy, F1, and MCC. It detected 39 of the 42 malignant cases and had zero false positives on this split. |
| **Overall winner for this dataset** | **SVM had the strongest held-out results** using F1 and MCC as the main comparison metrics. **Logistic Regression was the model selected by training-only cross-validation**, with mean F1 of 0.9640 versus 0.9610 for SVM. These outcomes describe two different evaluations. |

## Run the app

Use Python 3.13 to match the training environment. Run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Upload `test_data.csv` and select a classifier to see its six metrics, confusion matrix, and classification report. To reproduce training, run `python train_models.py --output-dir reproduced_run`.
