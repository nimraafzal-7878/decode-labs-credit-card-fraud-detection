
import pandas as pd

# Load Credit Card Fraud Detection dataset
df = pd.read_csv("creditcard.csv")

# Display first 5 rows
print("\n----- FIRST 5 ROWS -----")
print(df.head())

# Display dataset shape
print("\n----- DATASET SHAPE -----")
print(df.shape)

# Display column names
print("\n----- COLUMN NAMES -----")
print(df.columns.tolist())


# Step 3: Exploratory Data Analysis (EDA)

# Display dataset information
print("\n----- DATASET INFORMATION -----")
df.info()

# Check missing values
print("\n----- MISSING VALUES -----")
print(df.isnull().sum())

# Count genuine and fraudulent transactions
print("\n----- CLASS DISTRIBUTION -----")
print(df["Class"].value_counts())

# Calculate class percentages
print("\n----- CLASS PERCENTAGES -----")
print(df["Class"].value_counts(normalize=True) * 100)


# Step 4: Genuine vs Fraud Transactions

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 6))

# Original bar chart
sns.countplot(x="Class", data=df, ax=ax)

ax.set_title("Genuine vs Fraud Transactions")
ax.set_xlabel("Transaction Class")
ax.set_ylabel("Number of Transactions")
ax.set_xticks([0, 1], ["Genuine (0)", "Fraud (1)"])

# Get transaction counts
counts = df["Class"].value_counts()

# Display counts above bars
for i in [0, 1]:
    ax.text(i, counts[i] + 5000, f"{counts[i]:,}",
            ha="center")

# Add a small zoomed graph for Fraud
zoom = ax.inset_axes([0.58, 0.48, 0.30, 0.35])

zoom.bar([0], [counts[1]], color="orange")
zoom.set_ylim(0, 600)
zoom.set_xticks([0], ["Fraud (1)"])
zoom.set_title("Fraud - Zoomed View")

zoom.text(0, counts[1] + 15, str(counts[1]),
          ha="center")

plt.show()


# Step 5: Train/Test Split

from sklearn.model_selection import train_test_split

# Separate input features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display results
print("\n----- TRAIN/TEST SPLIT -----")

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())


# Step 6: Apply SMOTE on Training Data Only

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train, y_train
)

print("\n----- BEFORE SMOTE -----")
print(y_train.value_counts())

print("\n----- AFTER SMOTE -----")
print(y_train_smote.value_counts())

print("\nTraining Shape After SMOTE:")
print(X_train_smote.shape)


# Step 7: Logistic Regression Model Training

from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Create a Machine Learning Pipeline
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000))
])

# Train the model
print("\n----- TRAINING LOGISTIC REGRESSION -----")
lr_pipeline.fit(X_train, y_train)

# Predict transaction classes
y_pred_lr = lr_pipeline.predict(X_test)

# Predict fraud probabilities
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\n----- LOGISTIC REGRESSION RESULTS -----")

print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))


# Step 8: Random Forest Model Training

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

# Train model using SMOTE-balanced training data
print("\n----- TRAINING RANDOM FOREST -----")
rf_model.fit(X_train_smote, y_train_smote)

# Predict Genuine (0) or Fraud (1)
y_pred_rf = rf_model.predict(X_test)

# Predict Fraud risk scores
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluate Random Forest
print("\n----- RANDOM FOREST RESULTS -----")

print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))


# Step 9: Model Performance Comparison

comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Precision": [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_rf)
    ],
    "Recall": [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_rf)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, y_prob_lr),
        roc_auc_score(y_test, y_prob_rf)
    ]
})

print("\n----- MODEL COMPARISON -----")
print(comparison.round(4))


# Step 10: Random Forest Confusion Matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Calculate confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

print("\n----- RANDOM FOREST CONFUSION MATRIX -----")
print(cm_rf)

# Display confusion matrix graph
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=["Genuine (0)", "Fraud (1)"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()


# Step 11: Random Forest Threshold Tuning

import numpy as np
from sklearn.metrics import precision_recall_curve

# Split training data into sub-training and validation sets
X_subtrain, X_val, y_subtrain, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)

# Create a leakage-safe pipeline
rf_tuning_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])

# Train model on sub-training data
print("\n----- TRAINING FOR THRESHOLD TUNING -----")
rf_tuning_pipeline.fit(X_subtrain, y_subtrain)

# Predict fraud scores on validation data
val_prob = rf_tuning_pipeline.predict_proba(X_val)[:, 1]

# Calculate Precision and Recall at different thresholds
precision, recall, thresholds = precision_recall_curve(
    y_val, val_prob
)

# Choose highest Precision while keeping Recall >= 80%
valid_indices = np.where(recall[:-1] >= 0.80)[0]
best_index = valid_indices[
    np.argmax(precision[valid_indices])
]

best_threshold = thresholds[best_index]

# Predict using the selected threshold
val_pred = (val_prob >= best_threshold).astype(int)

print("\n----- THRESHOLD TUNING RESULTS -----")
print("Selected Threshold:", round(best_threshold, 4))
print("Validation Precision:",
      round(precision_score(y_val, val_pred), 4))
print("Validation Recall:",
      round(recall_score(y_val, val_pred), 4))


# Step 12: Random Forest Hyperparameter Tuning

from sklearn.model_selection import RandomizedSearchCV

# Use a smaller stratified sample for faster tuning
X_tune, _, y_tune, _ = train_test_split(
    X_subtrain,
    y_subtrain,
    train_size=0.30,
    random_state=42,
    stratify=y_subtrain
)

# Create pipeline
rf_search_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(
        random_state=42,
        n_jobs=1
    ))
])

# Hyperparameters to test
param_grid = {
    "model__n_estimators": [40, 70],
    "model__max_depth": [8, 12],
    "model__min_samples_leaf": [2, 5]
}

# Search for best settings
random_search = RandomizedSearchCV(
    estimator=rf_search_pipeline,
    param_distributions=param_grid,
    n_iter=4,
    cv=3,
    scoring="average_precision",
    random_state=42,
    n_jobs=-1
)

# Train and tune model
print("\n----- HYPERPARAMETER TUNING -----")
random_search.fit(X_tune, y_tune)

# Display best settings
print("\nBest Parameters:")
print(random_search.best_params_)

print("\nBest Cross-Validation PR-AUC:")
print(random_search.best_score_)

# Check tuned model on validation data
best_model = random_search.best_estimator_

val_predictions = best_model.predict(X_val)
val_probabilities = best_model.predict_proba(X_val)[:, 1]

print("\n----- TUNED MODEL VALIDATION -----")
print("Precision:", precision_score(y_val, val_predictions))
print("Recall:", recall_score(y_val, val_predictions))
print("ROC-AUC:", roc_auc_score(y_val, val_probabilities))


# Step 13: Threshold Selection for Tuned Model

import numpy as np
from sklearn.metrics import precision_recall_curve

# Predict fraud scores on validation data
tuned_val_prob = best_model.predict_proba(X_val)[:, 1]

# Calculate precision and recall at different thresholds
p_val, r_val, thresholds_val = precision_recall_curve(
    y_val, tuned_val_prob
)

# Select thresholds where recall is at least 80%
valid_indices = np.where(r_val[:-1] >= 0.80)[0]

# Find the highest precision among valid thresholds
best_idx = valid_indices[
    np.argmax(p_val[valid_indices])
]

# Get selected threshold
tuned_threshold = float(thresholds_val[best_idx])

# Make predictions using selected threshold
tuned_val_pred = (
    tuned_val_prob >= tuned_threshold
).astype(int)

# Display results
print("\n----- TUNED THRESHOLD RESULTS -----")
print("Selected Threshold:", round(tuned_threshold, 4))

print("Validation Precision:",
      precision_score(y_val, tuned_val_pred))

print("Validation Recall:",
      recall_score(y_val, tuned_val_pred))


# Step 14: Final Tuned Random Forest Model

from sklearn.base import clone
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# Create final model using best settings
final_model = clone(best_model)

# Train on complete training dataset
print("\n----- TRAINING FINAL MODEL -----")
final_model.fit(X_train, y_train)

# Predict fraud risk scores on test data
y_prob_final = final_model.predict_proba(X_test)[:, 1]

# Apply threshold selected in Step 13
y_pred_final = (
    y_prob_final >= tuned_threshold
).astype(int)

# Evaluate final model
print("\n----- FINAL MODEL RESULTS -----")

print("Precision:",
      precision_score(y_test, y_pred_final))

print("Recall:",
      recall_score(y_test, y_pred_final))

print("F1-Score:",
      f1_score(y_test, y_pred_final))

print("ROC-AUC:",
      roc_auc_score(y_test, y_prob_final))

print("Average Precision:",
      average_precision_score(y_test, y_prob_final))

print("\nFinal Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))


# Step 15: Final Model Comparison

# Create comparison table
final_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Tuned Random Forest"
    ],

    "Precision": [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_final)
    ],

    "Recall": [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_final)
    ],

    "F1-Score": [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_final)
    ],

    "ROC-AUC": [
        roc_auc_score(y_test, y_prob_lr),
        roc_auc_score(y_test, y_prob_rf),
        roc_auc_score(y_test, y_prob_final)
    ]
})

# Display comparison table
print("\n----- FINAL MODEL COMPARISON -----")
print(final_comparison.round(4))


# Step 16: Save Evaluation Graphs

from pathlib import Path
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

# Create a folder to save graphs
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)


# Graph 1: Final Model Confusion Matrix

fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_final,
    display_labels=["Genuine", "Fraud"],
    cmap="Blues",
    values_format="d",
    ax=ax
)

plt.title("Tuned Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
plt.close()


# Graph 2: ROC Curve Comparison

plt.figure(figsize=(8, 6))

model_scores = {
    "Logistic Regression": y_prob_lr,
    "Random Forest": y_prob_rf,
    "Tuned Random Forest": y_prob_final
}

for name, scores in model_scores.items():

    fpr, tpr, _ = roc_curve(y_test, scores)
    auc_value = roc_auc_score(y_test, scores)

    plt.plot(
        fpr, tpr,
        label=f"{name} (AUC={auc_value:.3f})"
    )

plt.plot([0, 1], [0, 1], "k--")

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()

plt.savefig(output_dir / "roc_curve.png", dpi=300)
plt.close()


# Graph 3: Precision-Recall Curve

plt.figure(figsize=(8, 6))

for name, scores in model_scores.items():

    precision, recall, _ = precision_recall_curve(
        y_test, scores
    )

    ap = average_precision_score(y_test, scores)

    plt.plot(
        recall, precision,
        label=f"{name} (AP={ap:.3f})"
    )

plt.title("Precision-Recall Curve Comparison")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()

plt.savefig(
    output_dir / "precision_recall_curve.png",
    dpi=300
)
plt.close()

print("\n----- GRAPHS SAVED SUCCESSFULLY -----")
print("confusion_matrix.png")
print("roc_curve.png")
print("precision_recall_curve.png")


# Step 17: Save Trained Model and Results

import joblib

# Save model comparison table as CSV
final_comparison.to_csv(
    output_dir / "model_comparison.csv",
    index=False
)


# Save the selected Random Forest model

joblib.dump(
    {
        "model": rf_model,
        "threshold": 0.5,
        "feature_names": X.columns.tolist()
    },
    output_dir / "fraud_detection_model.joblib"
)

# Display confirmation
print("\n----- MODEL AND RESULTS SAVED -----")
print("model_comparison.csv")
print("fraud_detection_model.joblib")
