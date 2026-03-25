import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# HOW TO USE THIS FUNCTION:
#
# run_model_validation("your_file.csv", "target_column")
#
# ARGUMENT 1 - "your_file.csv"
#   → The name of your dataset file (must be in same folder)
#   → Example: "patients.csv", "loans.csv", "data.csv"
#
# ARGUMENT 2 - "target_column"
#   → The column name that your AI is trying to predict
#   → Example: "disease", "approved", "price", "label"
#
# EXAMPLES:
#   run_model_validation("patients.csv", "disease")
#   run_model_validation("loans.csv", "approved")
#   run_model_validation("your_data.csv", "target")
#
# NOTE:
#   → Your CSV file and this script must be in the SAME folder
#   → Target column must exist in your CSV file
# ============================================================

def check_accuracy(y_true, y_pred):
    print("\n--- Accuracy Check ---")
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")
    if acc >= 0.80:
        print("PASS - Accuracy is acceptable!")
    else:
        print("FAIL - Accuracy too low! Model needs retraining.")
    return acc

def check_precision_recall_f1(y_true, y_pred):
    print("\n--- Precision, Recall, F1 Score ---")
    precision = precision_score(y_true, y_pred, average='weighted')
    recall    = recall_score(y_true, y_pred, average='weighted')
    f1        = f1_score(y_true, y_pred, average='weighted')
    print(f"Precision : {precision:.2f}")
    print(f"Recall    : {recall:.2f}")
    print(f"F1 Score  : {f1:.2f}")
    if f1 >= 0.75:
        print("PASS - F1 Score is good!")
    else:
        print("FAIL - F1 Score too low!")

def check_overfitting(train_acc, test_acc):
    print("\n--- Overfitting Check ---")
    print(f"Training Accuracy : {train_acc * 100:.2f}%")
    print(f"Testing Accuracy  : {test_acc * 100:.2f}%")
    gap = train_acc - test_acc
    if gap > 0.15:
        print("FAIL - Model is OVERFITTING! It memorized training data.")
    else:
        print("PASS - No significant overfitting detected!")

def check_confusion_matrix(y_true, y_pred, class_names):
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as confusion_matrix.png")

def run_model_validation(filepath, target_column):
    print("=============================")
    print("  AI V&V MODEL VALIDATOR     ")
    print("=============================")

    # Load dataset
    df = pd.read_csv(filepath)
    print(f"\nDataset loaded! Shape: {df.shape}")

    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Get class names
    class_names = [str(c) for c in sorted(y.unique())]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train model
    print("\nTraining model... please wait!")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred_test  = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test, y_pred_test)

    # Run all checks
    check_accuracy(y_test, y_pred_test)
    check_precision_recall_f1(y_test, y_pred_test)
    check_overfitting(train_acc, test_acc)
    check_confusion_matrix(y_test, y_pred_test, class_names)

    print("\n=============================")
    print("  MODEL VALIDATION COMPLETE!")
    print("=============================")

run_model_validation("your_data.csv", "target")
