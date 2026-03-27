import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix,
                              classification_report)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# AI V&V MODEL VALIDATOR - UPGRADED
# Based on: QB4AIRA (Document 2) + RPIA (Document 1)
#
# HOW TO USE:
#   run_model_validation("your_file.csv", "target_column")
#
# ARGUMENT 1 - "your_file.csv"
#   → Your dataset CSV file (must be in same folder)
#   → Example: "patients.csv", "loans.csv", "data.csv"
#
# ARGUMENT 2 - "target_column"
#   → The column your AI is trying to predict
#   → Example: "disease", "approved", "hired", "target"
#
# EXAMPLES:
#   run_model_validation("patients.csv", "disease")
#   run_model_validation("loans.csv",    "approved")
#   run_model_validation("your_data.csv","target")
#
# NOTE:
#   → CSV file and this script must be in the SAME folder
#   → Target column must exist in your CSV file
#   → Final report saved to: model_validation_report.txt
#   → Confusion matrix saved to: confusion_matrix.png
#   → Feature importance saved to: feature_importance.png
# ============================================================

# ─────────────────────────────────────────────
# ORIGINAL CHECKS (Steps 1-4)
# ─────────────────────────────────────────────

def check_accuracy(y_true, y_pred):
    """
    Basic accuracy check
    QB4AIRA — Reliability/Safety principle
    """
    print("\n--- [1] Accuracy Check ---")
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")
    if acc >= 0.80:
        print("PASS - Accuracy is acceptable!")
        return "PASS", acc
    else:
        print("FAIL - Accuracy too low! Model needs retraining.")
        return "FAIL", acc

def check_precision_recall_f1(y_true, y_pred):
    """
    Precision, Recall, F1 check
    QB4AIRA — Reliability/Safety + Fairness principle
    """
    print("\n--- [2] Precision, Recall, F1 Score ---")
    precision = precision_score(y_true, y_pred, average='weighted')
    recall    = recall_score(y_true, y_pred, average='weighted')
    f1        = f1_score(y_true, y_pred, average='weighted')
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    if f1 >= 0.75:
        print("PASS - F1 Score is good!")
        return "PASS", f1
    else:
        print("FAIL - F1 Score too low!")
        return "FAIL", f1

def check_overfitting(train_acc, test_acc):
    """
    Overfitting / Underfitting check
    QB4AIRA — Reliability/Safety principle
    RPIA Doc1 — Performance Degradation risk (PD)
    """
    print("\n--- [3] Overfitting Check ---")
    print(f"Training Accuracy : {train_acc * 100:.2f}%")
    print(f"Testing Accuracy  : {test_acc * 100:.2f}%")
    gap = train_acc - test_acc
    print(f"Gap               : {gap * 100:.2f}%")
    if gap > 0.15:
        print("FAIL - Model is OVERFITTING! Memorized training data.")
        print("  Fix: Add more data, use regularization, reduce complexity.")
        return "FAIL"
    elif test_acc < 0.60:
        print("FAIL - Model is UNDERFITTING! Accuracy too low.")
        print("  Fix: Use a more complex model or add more features.")
        return "FAIL"
    else:
        print("PASS - No significant overfitting or underfitting!")
        return "PASS"

def check_confusion_matrix(y_true, y_pred, class_names):
    """
    Confusion matrix — visual error analysis
    QB4AIRA — Transparency/Explainability principle
    """
    print("\n--- [4] Confusion Matrix ---")
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

    # Check for severely misclassified classes
    for i, cls in enumerate(class_names):
        total = cm[i].sum()
        correct = cm[i][i]
        if total > 0:
            cls_acc = correct / total
            if cls_acc < 0.60:
                print(f"  WARNING - Class '{cls}' accuracy is only "
                      f"{cls_acc*100:.1f}% — model struggles here!")
    return "PASS"

# ─────────────────────────────────────────────
# NEW CHECKS FROM QB4AIRA (Document 2)
# ─────────────────────────────────────────────

def check_explainability(model, feature_names):
    """
    QB4AIRA Principle: Transparency and Explainability
    Checks if model can explain its decisions via feature importance
    Source: QB4AIRA — explainability of system sub-category
    RPIA Doc1 — Risk 15: black box without documentation
    """
    print("\n--- [5] Explainability Check (QB4AIRA — Transparency Principle) ---")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("Feature importances (higher = more important for prediction):")
        for i in indices:
            bar = "#" * int(importances[i] * 40)
            print(f"  {feature_names[i]:<30} {importances[i]:.4f}  {bar}")

        # Save feature importance plot
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(importances)),
                importances[indices], color='steelblue')
        plt.xticks(range(len(importances)),
                   [feature_names[i] for i in indices], rotation=45)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        print("Feature importance chart saved as feature_importance.png")
        print("PASS - Model is explainable! (Not a black box)")
        return "PASS"
    else:
        print("FAIL - Model is a BLACK BOX!")
        print("  Cannot explain its decisions.")
        print("  Fix: Use RandomForest, DecisionTree, or add SHAP values.")
        return "FAIL"

def check_cross_validation(model, X, y):
    """
    QB4AIRA Principle: Reliability and Safety
    Cross validation — checks model consistency across different data splits
    Source: QB4AIRA — reliability and reproducibility sub-category
    """
    print("\n--- [6] Cross Validation Check (QB4AIRA — Reliability Principle) ---")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"CV Scores (5 folds): {[round(s, 4) for s in cv_scores]}")
    print(f"Mean Accuracy      : {cv_scores.mean():.4f}")
    print(f"Std Deviation      : {cv_scores.std():.4f}")

    if cv_scores.std() > 0.10:
        print("FAIL - High variance across folds!")
        print("  Model is unstable — results change too much with different data.")
        return "FAIL"
    elif cv_scores.mean() < 0.75:
        print("FAIL - Mean CV accuracy too low!")
        return "FAIL"
    else:
        print("PASS - Model is consistent and reliable across data splits!")
        return "PASS"

def check_per_class_fairness(y_true, y_pred, class_names):
    """
    QB4AIRA Principle: Fairness
    Checks if model performs equally well across all classes
    Source: QB4AIRA — unfair bias avoidance sub-category
    """
    print("\n--- [7] Per-Class Fairness Check (QB4AIRA — Fairness Principle) ---")
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )

    issues = []
    scores = []
    print("Per-class F1 scores:")
    for cls in class_names:
        if cls in report:
            f1 = report[cls]['f1-score']
            scores.append(f1)
            bar = "#" * int(f1 * 20)
            print(f"  Class '{cls}':{f1:.4f}  {bar}")
            if f1 < 0.60:
                issues.append(f"Class '{cls}' F1={f1:.2f} — model is unfair to this class!")

    if len(scores) > 1:
        fairness_gap = max(scores) - min(scores)
        print(f"\nFairness gap (max-min F1): {fairness_gap:.4f}")
        if fairness_gap > 0.20:
            issues.append(
                f"Large fairness gap ({fairness_gap:.2f}) — model treats classes unequally!"
            )

    if issues:
        for issue in issues:
            print(f"  FAIL - {issue}")
        return "FAIL"
    else:
        print("PASS - Model performs fairly across all classes!")
        return "PASS"

def check_model_reliability(model, X_test, y_test):
    """
    QB4AIRA Principle: Reliability and Safety
    Checks model prediction confidence and fallback safety
    Source: QB4AIRA — fallback plan and general safety sub-category
    RPIA Doc1 — Performance Degradation (PD) risk
    """
    print("\n--- [8] Reliability and Safety Check (QB4AIRA — Safety Principle) ---")

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        max_confidence = proba.max(axis=1)
        mean_conf = max_confidence.mean()
        low_conf_count = (max_confidence < 0.60).sum()
        low_conf_pct = (low_conf_count / len(X_test)) * 100

        print(f"Mean prediction confidence : {mean_conf:.4f}")
        print(f"Low confidence predictions : {low_conf_count} ({low_conf_pct:.1f}%)")

        if low_conf_pct > 20:
            print("FAIL - Too many uncertain predictions!")
            print("  Fix: Model needs more training data or better features.")
            return "FAIL"
        else:
            print("PASS - Model predictions are confident and reliable!")
            return "PASS"
    else:
        print("INFO - Model does not support probability scores.")
        return "PASS"

# ─────────────────────────────────────────────
# NEW CHECK FROM RPIA (Document 1)
# ─────────────────────────────────────────────

def check_rpia_model_risks(train_acc, test_acc, f1, cv_std):
    """
    RPIA Document 1 — Risk types mapped to model behavior:
    → PD  (Performance Degradation) — accuracy drop
    → Risk 9:  No audit logging / black box
    → Risk 14: Lack of stress testing
    → Risk 19: Model drift risk
    Source: RPIA Table 1 — Risks 9, 14, 19
    """
    print("\n--- [9] RPIA Model Risk Check (Document 1 — Table 1) ---")
    risks = []

    # PD Risk — performance degradation
    gap = train_acc - test_acc
    if gap > 0.15:
        risks.append(
            "RPIA Risk — Performance Degradation (PD) detected! "
            f"Train={train_acc*100:.1f}% vs Test={test_acc*100:.1f}%"
        )

    # Risk 14 — lack of stress testing (low F1)
    if f1 < 0.75:
        risks.append(
            f"RPIA Risk 14 — Low F1={f1:.2f}. Model not stress-tested enough!"
        )

    # Risk 19 — model drift risk (high CV variance)
    if cv_std > 0.10:
        risks.append(
            f"RPIA Risk 19 — High CV variance ({cv_std:.2f}). "
            f"Model may drift in production!"
        )

    # Risk 9 — auditability
    risks.append(
        "RPIA Risk 9 — Ensure model predictions are logged for audit trail!"
    )

    for r in risks:
        if "Ensure" in r:
            print(f"  INFO  - {r}")
        else:
            print(f"  FAIL  - {r}")

    hard_fails = [r for r in risks if "Ensure" not in r]
    if hard_fails:
        return "FAIL"
    else:
        print("PASS - No critical RPIA model risks detected!")
        return "PASS"

# ─────────────────────────────────────────────
# MAIN RUNNER + SUMMARY REPORT
# ─────────────────────────────────────────────

def run_model_validation(filepath, target_column):
    print("=" * 45)
    print("   AI V&V MODEL VALIDATOR (UPGRADED)   ")
    print("   QB4AIRA + RPIA aligned               ")
    print("=" * 45)

    # Load dataset
    df = pd.read_csv(filepath)
    print(f"\nDataset loaded  : {filepath}")
    print(f"Shape           : {df.shape}")
    print(f"Target column   : {target_column}")

    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    feature_names = list(X.columns)
    class_names   = [str(c) for c in sorted(y.unique())]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train model
    print("\nTraining model... please wait!")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred_test  = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    train_acc    = accuracy_score(y_train, y_pred_train)
    test_acc     = accuracy_score(y_test, y_pred_test)

    # ── Run all checks ──
    results = {}

    r1, acc    = check_accuracy(y_test, y_pred_test)
    results["Accuracy"]                  = r1

    r2, f1     = check_precision_recall_f1(y_test, y_pred_test)
    results["Precision/Recall/F1"]       = r2

    results["Overfitting"]               = check_overfitting(train_acc, test_acc)
    results["Confusion Matrix"]          = check_confusion_matrix(
                                               y_test, y_pred_test, class_names)
    results["Explainability (QB4AIRA)"]  = check_explainability(
                                               model, feature_names)

    # Cross validation needs fresh model
    cv_model   = RandomForestClassifier(random_state=42)
    r6         = check_cross_validation(cv_model, X, y)
    results["Cross Validation (QB4AIRA)"]= r6

    # Get CV std for RPIA check
    cv_scores  = cross_val_score(
        RandomForestClassifier(random_state=42), X, y, cv=5)
    cv_std     = cv_scores.std()

    results["Per-Class Fairness (QB4AIRA)"] = check_per_class_fairness(
                                               y_test, y_pred_test, class_names)
    results["Reliability/Safety (QB4AIRA)"] = check_model_reliability(
                                               model, X_test, y_test)
    results["RPIA Model Risk"]              = check_rpia_model_risks(
                                               train_acc, test_acc, f1, cv_std)

    # ── Summary ──
    print("\n" + "=" * 45)
    print("   MODEL VALIDATION SUMMARY REPORT     ")
    print("=" * 45)
    passed = 0
    failed = 0
    for check, result in results.items():
        symbol = "+" if result == "PASS" else "x"
        print(f"  [{symbol}] {check:<35} {result}")
        if result == "PASS":
            passed += 1
        else:
            failed += 1

    print(f"\nTotal checks : {len(results)}")
    print(f"Passed       : {passed}")
    print(f"Failed       : {failed}")

    if failed == 0:
        print("\nOVERALL: MODEL IS READY FOR DEPLOYMENT!")
    elif failed <= 2:
        print("\nOVERALL: MODEL NEEDS MINOR FIXES BEFORE DEPLOYMENT.")
    else:
        print("\nOVERALL: MODEL HAS SERIOUS ISSUES — DO NOT DEPLOY YET!")

    # ── Save report ──
    report_lines = [
        "AI V&V MODEL VALIDATION REPORT",
        f"File          : {filepath}",
        f"Target        : {target_column}",
        f"Train Accuracy: {train_acc*100:.2f}%",
        f"Test Accuracy : {test_acc*100:.2f}%",
        f"F1 Score      : {f1:.4f}",
        f"CV Std        : {cv_std:.4f}",
        f"Passed        : {passed}/{len(results)}",
        f"Failed        : {failed}/{len(results)}",
        ""
    ]
    for check, result in results.items():
        report_lines.append(f"{check}: {result}")

    with open("model_validation_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    print("\nFull report saved to : model_validation_report.txt")
    print("Confusion matrix     : confusion_matrix.png")
    print("Feature importance   : feature_importance.png")

    print("=" * 45)
    print("   MODEL VALIDATION COMPLETE!")
    print("=" * 45)

# --- RUN IT ---
run_model_validation("adult.csv", "income")