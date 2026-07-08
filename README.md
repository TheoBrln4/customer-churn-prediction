# Customer Churn Prediction

End-to-end Machine Learning project for predicting customer churn using **XGBoost**, with experiment tracking using **MLflow**, model explainability through **SHAP**, and business-oriented threshold optimization.

---

## Project Overview

Customer churn represents one of the most important business challenges for subscription-based companies.

The objective of this project is to predict whether a customer is likely to cancel their subscription based on their demographic information, subscribed services and billing history.

Rather than focusing only on model performance, this project also explores:

- experiment tracking with MLflow;
- business-oriented threshold optimization;
- model interpretability with SHAP.

---

## Dataset

This project uses the **Telco Customer Churn** dataset.

The dataset contains customer information such as:

- customer demographics;
- subscribed services;
- contract type;
- billing information;
- tenure;
- churn status.

Target variable:

```
Churn
```

- 0 → Customer stays
- 1 → Customer churns

The dataset can be downloaded from Kaggle:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

# Project Pipeline

```
EDA
        │
        ▼
Data preprocessing
        │
        ▼
Baseline Logistic Regression
        │
        ▼
XGBoost
        │
        ▼
MLflow Experiment Tracking
        │
        ▼
Threshold Optimization
        │
        ▼
SHAP Explainability
```

---

# Exploratory Data Analysis

The exploratory analysis focused on:

- missing values
- duplicated observations
- numerical distributions
- categorical distributions
- feature correlations
- churn distribution

Several business insights were identified.

For example:

- customers with **Month-to-month contracts** churn significantly more frequently;
- customers with **high monthly charges** are more likely to churn;
- customers with **long tenure** are much less likely to leave.

---

# Data Preprocessing

A complete Scikit-learn preprocessing pipeline was implemented.

### Numerical features

- Median imputation

### Categorical features

- Most frequent imputation
- One-Hot Encoding

The preprocessing is performed using:

- ColumnTransformer
- Pipeline

ensuring a reproducible workflow and preventing data leakage.

---

# Models

## Baseline

Logistic Regression was first trained to provide a simple and interpretable baseline.

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## XGBoost

A second model was built using XGBoost.

Main hyperparameters explored:

- learning_rate
- max_depth
- n_estimators
- subsample
- colsample_bytree

Several configurations were manually compared and tracked with MLflow.

---

# Experiment Tracking (MLflow)

All experiments are tracked using **MLflow**.

To launch the MLflow tracking interface:

```bash
mlflow ui
```

Then open:

```
http://localhost:5000
```

to explore and compare experiment runs.

Logged information includes:

### Parameters

- learning_rate
- max_depth
- n_estimators
- subsample
- colsample_bytree
- decision threshold

### Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Artifacts

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Trained model

---

# Business Threshold Optimization

By default, binary classifiers use a probability threshold of **0.5**.

However, the default threshold is not always optimal from a business perspective.

A simple business cost model was introduced:

| Error | Cost |
|--------|-----:|
| False Positive | €20 |
| False Negative | €60 |

The objective was to minimize the total business cost rather than maximizing a single Machine Learning metric.

Several thresholds were evaluated.

Lower thresholds significantly increased recall, allowing the model to detect more customers at risk of churn.

Higher thresholds reduced unnecessary retention campaigns but missed more actual churners.

The optimal threshold was therefore selected by minimizing the estimated business cost, we chose **0.25**.

This illustrates how model evaluation should be aligned with business objectives instead of relying solely on generic Machine Learning metrics.

---

# Model Explainability (SHAP)

SHAP was used to understand the model's predictions.

Two complementary visualizations were generated.

## Global explanation

The SHAP Summary Plot highlights the variables that globally influence churn prediction.

Main drivers include:

- Contract type
- Tenure
- Monthly Charges
- Online Security
- Fiber Optic Internet
- Tech Support

---

## Local explanation

The SHAP Waterfall Plot explains the prediction for an individual customer.

The explanation starts from the model's average prediction and shows how each feature pushes the prediction toward a higher or lower churn probability.

---

# Technologies

- Python
- Pandas
- Scikit-learn
- XGBoost
- MLflow
- SHAP

---

# Repository Structure

```
.
├── data/
├── notebook/
│   └── eda.ipynb
├── src/
│   ├── train_baseline.py
│   ├── train_xgboost.py
│   └── explain_shap.py
├── images/
├── .gitignore
└── README.md
```

---

# Future Improvements

- Hyperparameter optimization using Optuna
- Cross-validation
- Model deployment with FastAPI
- Drift monitoring
- Feature store integration

---

# Key Skills Demonstrated

- Exploratory Data Analysis
- Feature Engineering
- Machine Learning Pipelines
- Binary Classification
- XGBoost
- Experiment Tracking with MLflow
- Model Explainability with SHAP
- Business-oriented Decision Threshold Optimization
