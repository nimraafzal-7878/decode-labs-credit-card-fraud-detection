# Credit Card Fraud Detection Using Machine Learning

A supervised machine learning project designed to identify fraudulent credit card transactions using classification algorithms, class imbalance handling, and model optimization techniques.

The project implements Logistic Regression and Random Forest classifiers, addresses class imbalance using the Synthetic Minority Over-sampling Technique (SMOTE), and evaluates model performance using metrics suitable for highly imbalanced datasets.

## Project Overview

Credit card fraud detection is a challenging classification problem because fraudulent transactions represent only a small proportion of total financial transactions.

Traditional classification models may favor the majority class, resulting in high overall accuracy but poor fraud detection performance.

This project develops a supervised machine learning pipeline that incorporates exploratory data analysis, SMOTE-based resampling, classification model training, hyperparameter tuning, threshold optimization, and performance evaluation.

The primary objective is to identify fraudulent transactions while reducing unnecessary false-positive alerts.

## Project Objectives

- Analyze the distribution of genuine and fraudulent credit card transactions.
- Examine dataset structure, feature types, and missing values.
- Handle severe class imbalance using SMOTE.
- Develop Logistic Regression and Random Forest classification models.
- Optimize Random Forest hyperparameters using cross-validation.
- Investigate the effect of classification threshold adjustment.
- Evaluate models using Precision, Recall, F1-Score, ROC-AUC, and Average Precision.
- Visualize performance using confusion matrices and evaluation curves.
- Compare model configurations and select a suitable classifier.
- Save the selected trained model and evaluation results.

## Dataset Description

**Dataset:** Credit Card Fraud Detection

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains European credit card transactions recorded over approximately two days.

### Dataset Summary

| Characteristic | Value |
|---|---:|
| Total Transactions | 284,807 |
| Total Columns | 31 |
| Input Features | 30 |
| Target Variable | Class |
| Genuine Transactions (Class 0) | 284,315 |
| Fraudulent Transactions (Class 1) | 492 |
| Genuine Transaction Percentage | 99.827% |
| Fraudulent Transaction Percentage | 0.173% |
| Missing Values | 0 |

### Feature Description

- **Time:** Seconds elapsed between each transaction and the first recorded transaction.
- **V1–V28:** Anonymized numerical features obtained through Principal Component Analysis (PCA).
- **Amount:** Transaction amount.
- **Class:** Binary target variable, where `0` represents a genuine transaction and `1` represents a fraudulent transaction.

The significant imbalance between genuine and fraudulent transactions makes this dataset suitable for evaluating minority-class detection techniques.

## Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Classification, model selection, and evaluation |
| Imbalanced-learn | SMOTE and imbalance-aware pipelines |
| Joblib | Model serialization and storage |

## Methodology

### 1. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the structure and characteristics of the dataset.

The analysis included:

- Examining the first five records.
- Identifying dataset dimensions and feature names.
- Inspecting feature data types.
- Checking missing values.
- Analyzing class distribution.
- Visualizing genuine and fraudulent transaction frequencies.

The analysis confirmed that fraudulent transactions account for approximately **0.173%** of the dataset.

### 2. Train-Test Split

The dataset was divided using an **80:20 stratified train-test split**.

| Dataset | Transactions |
|---|---:|
| Training Set | 227,845 |
| Testing Set | 56,962 |

Stratified sampling preserved approximately the same fraud-to-genuine distribution in both subsets.

The test dataset was excluded from model training and SMOTE resampling.

### 3. Class Imbalance Handling Using SMOTE

The **Synthetic Minority Over-sampling Technique (SMOTE)** was implemented to address the imbalance between genuine and fraudulent transactions.

SMOTE generates synthetic minority-class observations by interpolating between existing minority-class samples.

**Training data before SMOTE:**

| Class | Transactions |
|---|---:|
| Genuine (0) | 227,451 |
| Fraud (1) | 394 |

**Training data after SMOTE:**

| Class | Transactions |
|---|---:|
| Genuine (0) | 227,451 |
| Fraud (1) | 227,451 |

The balanced training dataset contained **454,902 samples**.

SMOTE was applied exclusively to training data. During cross-validation, it was integrated into an `imblearn.pipeline.Pipeline` to prevent resampling leakage between training and validation folds.

### 4. Model Development

Two supervised machine learning algorithms were implemented.

#### Logistic Regression

Logistic Regression was used as the baseline classification model.

The training pipeline consisted of:

1. StandardScaler for feature standardization.
2. SMOTE for minority-class oversampling.
3. Logistic Regression for binary classification.

The model was evaluated using class predictions and fraud probability scores on the test dataset.

#### Random Forest

Random Forest was implemented as an ensemble learning algorithm combining multiple decision trees.

The initial model was configured using the following hyperparameters:

| Hyperparameter | Value |
|---|---:|
| `n_estimators` | 50 |
| `max_depth` | 12 |
| `min_samples_leaf` | 5 |
| `random_state` | 42 |

The model was trained using SMOTE-balanced training data and evaluated on the original test dataset.

### 5. Model Evaluation

The models were evaluated using the following metrics:

- **Precision:** Proportion of predicted fraudulent transactions that were actually fraudulent.
- **Recall:** Proportion of actual fraudulent transactions successfully detected.
- **F1-Score:** Harmonic mean of Precision and Recall.
- **ROC-AUC:** Ability to distinguish fraudulent transactions from genuine transactions based on prediction scores across thresholds.
- **Average Precision:** Summary of model performance across the Precision-Recall curve.
- **Confusion Matrix:** Distribution of true negatives, false positives, false negatives, and true positives.

Accuracy was not used as the primary evaluation metric because the extreme class imbalance could produce misleading results.

### 6. Hyperparameter Optimization

Random Forest hyperparameter tuning was performed using **RandomizedSearchCV** with three-fold cross-validation on a stratified subset of the training data.

The search considered variations in:

- Number of decision trees (`n_estimators`)
- Maximum tree depth (`max_depth`)
- Minimum samples per leaf (`min_samples_leaf`)

Average Precision was used as the optimization metric.

**Best Hyperparameters:**

| Hyperparameter | Selected Value |
|---|---:|
| `n_estimators` | 70 |
| `max_depth` | 12 |
| `min_samples_leaf` | 2 |

The best cross-validation Average Precision score was approximately **0.8316**.

### 7. Classification Threshold Optimization

Classification threshold optimization was conducted using a separate validation subset.

The objective was to maximize Precision while maintaining a minimum Recall of **80%**.

For the tuned Random Forest, the selected threshold was **0.2724**.

Validation performance at the selected threshold:

| Metric | Value |
|---|---:|
| Precision | 0.4969 |
| Recall | 0.8081 |

Reducing the classification threshold increased fraud detection sensitivity but also resulted in more false-positive predictions.

## Model Performance and Results

Three model configurations were evaluated:

1. Logistic Regression
2. Random Forest
3. Tuned Random Forest

### Performance Comparison

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.0578 | 0.9184 | 0.1088 | 0.9708 |
| **Random Forest** | **0.6014** | 0.8776 | **0.7137** | **0.9840** |
| Tuned Random Forest | 0.2189 | 0.8980 | 0.3520 | 0.9798 |

The original Random Forest achieved the highest Precision, F1-Score, and ROC-AUC among the evaluated configurations.

The tuned Random Forest achieved slightly higher Recall but generated substantially more false-positive predictions.

### Original Random Forest Confusion Matrix

| Prediction Outcome | Count |
|---|---:|
| True Negatives (TN) | 56,807 |
| False Positives (FP) | 57 |
| False Negatives (FN) | 12 |
| True Positives (TP) | 86 |

The original Random Forest correctly detected **86 of the 98 fraudulent transactions** in the test set, with 57 false-positive predictions.

### Tuned Random Forest Confusion Matrix

| Prediction Outcome | Count |
|---|---:|
| True Negatives (TN) | 56,550 |
| False Positives (FP) | 314 |
| False Negatives (FN) | 10 |
| True Positives (TP) | 88 |

The tuned Random Forest detected two additional fraudulent transactions compared with the original Random Forest but generated 257 additional false-positive predictions.

These results demonstrate the trade-off between fraud detection sensitivity and the frequency of unnecessary fraud alerts.

### Precision-Recall Curve Analysis

The Precision-Recall curves provided an additional perspective on model ranking performance across different classification thresholds.

Approximate Average Precision scores from the generated evaluation graph were:

| Model | Average Precision |
|---|---:|
| Logistic Regression | 0.724 |
| Random Forest | 0.811 |
| Tuned Random Forest | 0.836 |

Although the tuned Random Forest achieved the highest Average Precision, its selected operating threshold produced lower Precision and F1-Score than the original Random Forest.

This highlights the importance of distinguishing overall ranking quality from performance at a specific classification threshold.

## Final Model Selection

Three model configurations were evaluated: Logistic Regression, Random Forest, and Tuned Random Forest.

The **original Random Forest was selected as the preferred final model** based on its stronger Precision (**60.14%**), F1-Score (**0.7137**), and ROC-AUC (**0.9840**) in the exploratory test-set comparison.

Although the tuned Random Forest achieved slightly higher Recall (**89.80%**), it produced substantially more false-positive predictions.

The original Random Forest offered a more favorable balance between fraud detection performance and false-positive alerts under the selected evaluation criteria.

The selected classifier is saved as:

`results/fraud_detection_model.joblib`

The saved model uses a classification threshold of **0.50**.

**Model Selection Note:** The model was selected after comparing performance on the test dataset. Therefore, the reported results should be considered exploratory rather than completely independent final validation. Additional assessment using a separate unseen dataset is recommended before real-world deployment.

## Project Structure

```text
decodslab_task2/
│
├── main.py
├── creditcard.csv
├── README.md
│
└── results/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── model_comparison.csv
    └── fraud_detection_model.joblib
```
## Installation

### Prerequisites

- Python 3.10 or later
- pip package manager
- Credit Card Fraud Detection dataset

Install the required libraries using the following command:

`python -m pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib`

## How to Run

1. Download the Credit Card Fraud Detection dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Extract `creditcard.csv` into the project directory.
3. Install the required Python libraries.
4. Open the project folder in VS Code or a terminal.
5. Execute the main script using:

   `python main.py`

The script executes the workflow, including data exploration, SMOTE-based resampling, model training, hyperparameter tuning, threshold optimization, and performance evaluation.

Generated visualizations and trained model artifacts are saved in the `results` directory.

## Results and Visualizations

### Confusion Matrix

**File:** `results/confusion_matrix.png`

Displays the tuned Random Forest model's correct and incorrect classification outcomes, including true positives, true negatives, false positives, and false negatives.

### ROC Curve Comparison

**File:** `results/roc_curve.png`

Compares the ROC-AUC performance of Logistic Regression, Random Forest, and Tuned Random Forest.

### Precision-Recall Curve

**File:** `results/precision_recall_curve.png`

Illustrates the relationship between Precision and Recall across different classification thresholds.

### Model Comparison

**File:** `results/model_comparison.csv`

Contains Precision, Recall, F1-Score, and ROC-AUC results for all three model configurations.

### Saved Machine Learning Model

**File:** `results/fraud_detection_model.joblib`

Stores the selected Random Forest classifier, classification threshold, and required input feature names.

The saved model uses the original Random Forest configuration with a classification threshold of `0.50`.

## Limitations

- Fraudulent transactions represent a very small proportion of the dataset.
- SMOTE-generated observations are synthetic and may not fully represent real-world fraud patterns.
- PCA-transformed features limit the interpretation of individual predictors.
- Classification performance depends on model configuration and decision thresholds.
- Hyperparameter tuning used a subset of the training data, which may limit the parameter search.
- Model selection was informed by test-set comparisons, so the reported performance is exploratory rather than independently validated.
- Additional validation, monitoring, and testing on recent real-world transactions would be necessary before production deployment.

## Conclusion

This project demonstrates a supervised machine learning approach to credit card fraud detection using Logistic Regression, Random Forest, SMOTE, and model optimization techniques.

The original Random Forest classifier achieved the highest Precision, F1-Score, and ROC-AUC among the evaluated configurations, offering a favorable balance between fraud detection capability and false-positive reduction.

Although the tuned Random Forest achieved slightly higher Recall and Average Precision, its selected classification threshold produced substantially more false-positive predictions.

The original Random Forest was therefore selected as the preferred model for saving and further evaluation.

Overall, the project highlights the importance of class imbalance handling, appropriate evaluation metrics, threshold optimization, and careful model selection rather than relying solely on classification accuracy.