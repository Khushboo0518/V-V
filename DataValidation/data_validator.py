import pandas as pd
import numpy as np

# ============================================================
# AI V&V DATA VALIDATOR 
# Based on: QB4AIRA (Document 2) + RPIA (Document 1)
#
# HOW TO USE:
#   run_data_validation("your_file.csv", "target_column")
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
#   run_data_validation("patients.csv", "disease")
#   run_data_validation("loans.csv",    "approved")
#   run_data_validation("your_data.csv","target")
#
# NOTE:
#   → CSV file and this script must be in the SAME folder
#   → Target column must exist in your CSV file
#   → Final report saved to: data_validation_report.txt
# ============================================================

# ─────────────────────────────────────────────
# ORIGINAL CHECKS (Steps 1-4)
# ─────────────────────────────────────────────

def check_missing_values(df):
    print("\n--- [1] Missing Values Check ---")
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    result = pd.DataFrame({
        'Missing Count': missing,
        'Percent %': percent.round(2)
    })
    print(result)
    if missing.sum() == 0:
        print("PASS - No missing values found!")
        return "PASS"
    else:
        print("FAIL - Missing values detected! Consider imputation or removal.")
        return "FAIL"

def check_duplicates(df):
    print("\n--- [2] Duplicate Rows Check ---")
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicates}")
    if duplicates == 0:
        print("PASS - No duplicates!")
        return "PASS"
    else:
        print(f"FAIL - {duplicates} duplicate rows detected! Remove before training.")
        return "FAIL"

def check_data_types(df):
    print("\n--- [3] Data Types Check ---")
    print(df.dtypes)
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"WARNING - Non-numeric columns found: {non_numeric}")
        print("  These need encoding before model training!")
    else:
        print("PASS - All columns are numeric!")
    return "PASS"

def check_data_balance(df, target_column):
    print("\n--- [4] Data Balance Check ---")
    balance = df[target_column].value_counts()
    print(balance)
    min_class = balance.min()
    max_class = balance.max()
    ratio = min_class / max_class
    print(f"Balance ratio: {ratio:.2f} (min/max class size)")
    if ratio > 0.4:
        print("PASS - Data is reasonably balanced!")
        return "PASS"
    else:
        print("FAIL - Data is imbalanced! Bias risk detected!")
        print("  Fix: Use oversampling (SMOTE) or undersampling.")
        return "FAIL"

# ─────────────────────────────────────────────
# NEW CHECKS FROM QB4AIRA (Document 2)
# ─────────────────────────────────────────────

def check_data_privacy(df):
    """
    QB4AIRA Principle: Privacy and Security
    Checks if dataset contains PII (Personally Identifiable Information)
    Source: QB4AIRA — Privacy/Security category
    """
    print("\n--- [5] Privacy Check (QB4AIRA — Privacy Principle) ---")
    sensitive_keywords = [
        'name', 'email', 'phone', 'mobile', 'address',
        'ssn', 'dob', 'birthdate', 'password', 'credit',
        'medical', 'salary', 'income', 'gender', 'race',
        'religion', 'passport', 'aadhar', 'pan', 'ip'
    ]
    found = []
    for col in df.columns:
        for keyword in sensitive_keywords:
            if keyword in col.lower():
                found.append(col)
                break

    if found:
        print(f"WARNING - Sensitive/PII columns detected: {found}")
        print("FAIL - Privacy risk! Apply masking, encryption or anonymization.")
        return "FAIL"
    else:
        print("PASS - No obvious PII columns found!")
        return "PASS"

def check_data_bias(df, target_column):
    """
    QB4AIRA Principle: Fairness
    Checks if dataset contains columns that may introduce bias
    Source: QB4AIRA — Fairness category (unfair bias avoidance)
    """
    print("\n--- [6] Bias Risk Check (QB4AIRA — Fairness Principle) ---")
    bias_keywords = [
        'gender', 'sex', 'race', 'ethnicity', 'religion',
        'nationality', 'age', 'disability', 'caste', 'color'
    ]
    found = []
    for col in df.columns:
        for keyword in bias_keywords:
            if keyword in col.lower():
                found.append(col)
                break

    if found:
        print(f"WARNING - Potential bias-risk columns found: {found}")
        print("FAIL - Fairness risk! Review if these columns should be used.")
        print("  These protected attributes may cause discriminatory predictions.")
        return "FAIL"
    else:
        print("PASS - No obvious bias-risk columns found!")
        return "PASS"

def check_data_consistency(df):
    """
    QB4AIRA Principle: Reliability and Safety
    Checks for statistical anomalies and outliers in numeric columns
    Source: QB4AIRA — Reliability/accuracy category
    """
    print("\n--- [7] Data Consistency Check (QB4AIRA — Reliability Principle) ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    issues = []

    for col in numeric_cols:
        mean = df[col].mean()
        std  = df[col].std()
        if std == 0:
            issues.append(f"{col} has zero variance (constant column — useless!)")
            continue
        outliers = df[(df[col] > mean + 3*std) |
                      (df[col] < mean - 3*std)].shape[0]
        if outliers > 0:
            pct = round((outliers / len(df)) * 100, 2)
            issues.append(f"{col}: {outliers} outliers ({pct}% of data)")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  WARNING - {issue}")
        print("FAIL - Data consistency issues detected!")
        return "FAIL"
    else:
        print("PASS - No outliers or constant columns detected!")
        return "PASS"

def check_data_sufficiency(df, target_column):
    """
    QB4AIRA Principle: Reliability and Safety
    Checks if there is enough data per class for reliable training
    Source: QB4AIRA — accuracy/data quality sub-category
    """
    print("\n--- [8] Data Sufficiency Check (QB4AIRA — Reliability Principle) ---")
    total_rows = len(df)
    num_classes = df[target_column].nunique()
    num_features = df.shape[1] - 1

    print(f"Total rows     : {total_rows}")
    print(f"Total features : {num_features}")
    print(f"Total classes  : {num_classes}")
    print(f"Rows per class : ~{total_rows // num_classes}")

    issues = []
    if total_rows < 100:
        issues.append("Less than 100 rows — very risky for training!")
    if total_rows < num_features * 10:
        issues.append("Too few rows vs features — risk of overfitting!")
    per_class = df[target_column].value_counts()
    for cls, count in per_class.items():
        if count < 20:
            issues.append(f"Class '{cls}' has only {count} samples — too few!")

    if issues:
        for issue in issues:
            print(f"  FAIL - {issue}")
        return "FAIL"
    else:
        print("PASS - Sufficient data available for training!")
        return "PASS"

def check_feature_correlation(df, target_column):
    """
    QB4AIRA Principle: Transparency and Explainability
    Checks which features are most correlated with target
    Source: QB4AIRA — explainability of system sub-category
    """
    print("\n--- [9] Feature Correlation Check (QB4AIRA — Transparency Principle) ---")
    numeric_df = df.select_dtypes(include=[np.number])
    if target_column not in numeric_df.columns:
        print("INFO - Target column is not numeric, skipping correlation.")
        return "PASS"

    correlations = numeric_df.corr()[target_column].drop(target_column)
    correlations = correlations.abs().sort_values(ascending=False)

    print("Feature correlations with target:")
    for feat, corr in correlations.items():
        bar = "#" * int(corr * 20)
        print(f"  {feat:<30} {corr:.4f}  {bar}")

    weak = correlations[correlations < 0.05]
    if len(weak) > 0:
        print(f"\nWARNING - {len(weak)} features have very weak correlation:")
        print(f"  {list(weak.index)} — consider removing!")
    else:
        print("\nPASS - All features show meaningful correlation!")
    return "PASS"

# ─────────────────────────────────────────────
# NEW CHECK FROM RPIA (Document 1)
# ─────────────────────────────────────────────

def check_rpia_data_risk(df, target_column):
    """
    RPIA Document — Risk type: Data Risk
    Checks if data presents any of the RPIA-defined data risks:
    → Risk 5: Test data very different from production data
    → Risk 6: Biased or inaccurate training data
    → Risk 8: Lack of proper data sensitivity tagging
    Source: RPIA Table 1 — Risks 5, 6, 8
    """
    print("\n--- [10] RPIA Data Risk Check (Document 1 — Table 1) ---")
    risks_found = []

    # Risk 5 — check data spread (proxy for production mismatch)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        skewness = df[col].skew()
        if abs(skewness) > 2:
            risks_found.append(
                f"Risk 5 — '{col}' is highly skewed ({skewness:.2f}). "
                f"May not represent real-world distribution!"
            )

    # Risk 6 — check class balance again from RPIA perspective
    balance = df[target_column].value_counts()
    ratio = balance.min() / balance.max()
    if ratio < 0.3:
        risks_found.append(
            f"Risk 6 — Severely imbalanced classes (ratio={ratio:.2f}). "
            f"Training data is biased!"
        )

    # Risk 8 — check for sensitive column tagging
    sensitive_keywords = [
        'name','email','phone','ssn','medical',
        'salary','gender','race','password'
    ]
    untagged = []
    for col in df.columns:
        for kw in sensitive_keywords:
            if kw in col.lower():
                untagged.append(col)
                break
    if untagged:
        risks_found.append(
            f"Risk 8 — Sensitive columns not tagged/protected: {untagged}"
        )

    if risks_found:
        for r in risks_found:
            print(f"  FAIL - {r}")
        return "FAIL"
    else:
        print("PASS - No RPIA data risks detected!")
        return "PASS"

# ─────────────────────────────────────────────
# MAIN RUNNER + SUMMARY REPORT
# ─────────────────────────────────────────────

def run_data_validation(filepath, target_column):
    print("=" * 45)
    print("   AI V&V DATA VALIDATOR (UPGRADED)    ")
    print("   QB4AIRA + RPIA aligned               ")
    print("=" * 45)

    df = pd.read_csv(filepath)
    print(f"\nDataset loaded  : {filepath}")
    print(f"Shape           : {df.shape}")
    print(f"Target column   : {target_column}")

    # Run all checks and collect results
    results = {}
    results["Missing Values"]       = check_missing_values(df)
    results["Duplicates"]           = check_duplicates(df)
    results["Data Types"]           = check_data_types(df)
    results["Data Balance"]         = check_data_balance(df, target_column)
    results["Privacy (QB4AIRA)"]    = check_data_privacy(df)
    results["Bias Risk (QB4AIRA)"]  = check_data_bias(df, target_column)
    results["Consistency (QB4AIRA)"]= check_data_consistency(df)
    results["Sufficiency (QB4AIRA)"]= check_data_sufficiency(df, target_column)
    results["Correlation (QB4AIRA)"]= check_feature_correlation(df, target_column)
    results["RPIA Data Risk"]       = check_rpia_data_risk(df, target_column)

    # Print summary
    print("\n" + "=" * 45)
    print("   VALIDATION SUMMARY REPORT           ")
    print("=" * 45)
    passed = 0
    failed = 0
    for check, result in results.items():
        status = "PASS" if result == "PASS" else "FAIL"
        symbol = "+" if status == "PASS" else "x"
        print(f"  [{symbol}] {check:<30} {status}")
        if status == "PASS":
            passed += 1
        else:
            failed += 1

    print(f"\nTotal checks : {len(results)}")
    print(f"Passed       : {passed}")
    print(f"Failed       : {failed}")

    if failed == 0:
        print("\nOVERALL: DATA IS READY FOR TRAINING!")
    elif failed <= 3:
        print("\nOVERALL: DATA NEEDS MINOR FIXES BEFORE TRAINING.")
    else:
        print("\nOVERALL: DATA HAS SERIOUS ISSUES — DO NOT TRAIN YET!")

    # Save report to file
    import sys
    report_lines = []
    report_lines.append("AI V&V DATA VALIDATION REPORT")
    report_lines.append(f"File    : {filepath}")
    report_lines.append(f"Target  : {target_column}")
    report_lines.append(f"Passed  : {passed}/{len(results)}")
    report_lines.append(f"Failed  : {failed}/{len(results)}")
    for check, result in results.items():
        report_lines.append(f"{check}: {result}")
    with open("data_validation_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    print("\nFull report saved to: data_validation_report.txt")

    print("=" * 45)
    print("   VALIDATION COMPLETE!")
    print("=" * 45)

# --- RUN IT ---
run_data_validation("adult.csv", "income")
