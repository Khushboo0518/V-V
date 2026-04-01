from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os, io, base64, json, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

def pass_fail(condition, pass_msg, fail_msg):
    return {"status": "PASS", "msg": pass_msg} if condition else {"status": "FAIL", "msg": fail_msg}

# ─────────────────────────────────────────────
# MODULE 1 — DATA VALIDATION
# ─────────────────────────────────────────────

def run_data_validation(df, target_column):
    results = []
    charts  = []

    # 1. Missing Values
    missing = df.isnull().sum()
    pct     = (missing / len(df) * 100).round(2)
    fig, ax = plt.subplots(figsize=(8, 3))
    colors  = ['#e74c3c' if v > 0 else '#2ecc71' for v in missing.values]
    ax.barh(missing.index, pct.values, color=colors)
    ax.set_xlabel('Missing %')
    ax.set_title('Missing Values per Column')
    for i, v in enumerate(pct.values):
        ax.text(v + 0.1, i, f'{v:.1f}%', va='center', fontsize=8)
    plt.tight_layout()
    charts.append({"title": "Missing Values", "img": fig_to_base64(fig)})
    results.append({
        "check": "Missing Values",
        **pass_fail(missing.sum() == 0,
                    "No missing values found!",
                    f"{missing.sum()} missing values detected. Consider imputation."),
        "detail": missing[missing > 0].to_dict()
    })

    # 2. Duplicates
    dups = df.duplicated().sum()
    results.append({
        "check": "Duplicate Rows",
        **pass_fail(dups == 0,
                    "No duplicate rows!",
                    f"{dups} duplicate rows found. Remove before training."),
        "detail": {"duplicates": int(dups)}
    })

    # 3. Class Balance
    balance = df[target_column].value_counts()
    ratio   = float(balance.min() / balance.max())
    fig, ax = plt.subplots(figsize=(6, 3))
    colors  = sns.color_palette("Set2", len(balance))
    ax.bar([str(c) for c in balance.index], balance.values, color=colors)
    ax.set_title(f'Class Distribution — {target_column}')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    for i, v in enumerate(balance.values):
        ax.text(i, v + 0.5, str(v), ha='center', fontsize=9)
    plt.tight_layout()
    charts.append({"title": "Class Balance", "img": fig_to_base64(fig)})
    results.append({
        "check": "Class Balance",
        **pass_fail(ratio > 0.4,
                    f"Balanced! Ratio={ratio:.2f}",
                    f"Imbalanced! Ratio={ratio:.2f}. Bias risk!"),
        "detail": balance.to_dict()
    })

    # 4. Privacy — PII columns
    pii_kw  = ['name','email','phone','ssn','dob','password','credit',
                'salary','gender','race','religion','passport','aadhar','ip']
    pii_found = [c for c in df.columns if any(k in c.lower() for k in pii_kw)]
    results.append({
        "check": "Privacy (PII Check)",
        **pass_fail(len(pii_found) == 0,
                    "No PII columns found!",
                    f"PII columns found: {pii_found}. Anonymize them!"),
        "detail": {"pii_columns": pii_found}
    })

    # 5. Bias risk columns
    bias_kw   = ['gender','sex','race','ethnicity','religion','nationality',
                 'age','disability','caste','color']
    bias_found= [c for c in df.columns if any(k in c.lower() for k in bias_kw)]
    results.append({
        "check": "Bias Risk (Fairness)",
        **pass_fail(len(bias_found) == 0,
                    "No bias-risk columns found!",
                    f"Bias-risk columns: {bias_found}. Review usage!"),
        "detail": {"bias_columns": bias_found}
    })

    # 6. Outliers
    numeric = df.select_dtypes(include=[np.number])
    outlier_info = {}
    for col in numeric.columns:
        mean, std = df[col].mean(), df[col].std()
        if std > 0:
            n = int(((df[col] > mean+3*std)|(df[col] < mean-3*std)).sum())
            if n > 0:
                outlier_info[col] = n
    fig, ax = plt.subplots(figsize=(8, 3))
    if numeric.shape[1] > 0:
        numeric.iloc[:, :min(6, numeric.shape[1])].boxplot(ax=ax, vert=False)
    ax.set_title('Outlier Detection (Box Plot)')
    plt.tight_layout()
    charts.append({"title": "Outliers", "img": fig_to_base64(fig)})
    results.append({
        "check": "Outlier / Consistency",
        **pass_fail(len(outlier_info) == 0,
                    "No major outliers found!",
                    f"Outliers in: {list(outlier_info.keys())}"),
        "detail": outlier_info
    })

    # 7. Data sufficiency
    total   = len(df)
    n_class = df[target_column].nunique()
    n_feat  = df.shape[1] - 1
    suf_ok  = total >= 100 and total >= n_feat * 10
    results.append({
        "check": "Data Sufficiency",
        **pass_fail(suf_ok,
                    f"{total} rows, {n_feat} features — sufficient!",
                    f"Only {total} rows for {n_feat} features. Risk of overfitting!"),
        "detail": {"rows": total, "features": n_feat, "classes": n_class}
    })

    # 8. Correlation heatmap
    if numeric.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(7, 5))
        corr = numeric.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    ax=ax, mask=mask, linewidths=0.5, annot_kws={"size": 8})
        ax.set_title('Feature Correlation Heatmap')
        plt.tight_layout()
        charts.append({"title": "Correlation Heatmap", "img": fig_to_base64(fig)})

    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = len(results) - passed
    if   failed == 0:      overall = "DATA IS READY FOR TRAINING!"
    elif failed <= 3:      overall = "DATA NEEDS MINOR FIXES."
    else:                  overall = "SERIOUS DATA ISSUES — DO NOT TRAIN YET!"

    return {"results": results, "charts": charts,
            "passed": passed, "failed": failed, "overall": overall,
            "total": len(results)}

# ─────────────────────────────────────────────
# MODULE 2 — MODEL VALIDATION
# ─────────────────────────────────────────────

def run_model_validation(df, target_column):
    results = []
    charts  = []

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X = pd.get_dummies(X, drop_first=True)
    feature_names = list(X.columns)
    class_names   = [str(c) for c in sorted(y.unique())]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred      = model.predict(X_test)
    y_pred_tr   = model.predict(X_train)
    train_acc   = accuracy_score(y_train, y_pred_tr)
    test_acc    = accuracy_score(y_test, y_pred)
    precision   = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall      = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1          = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # 1. Accuracy gauge chart
    fig, ax = plt.subplots(figsize=(5, 3))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values  = [test_acc, precision, recall, f1]
    colors  = ['#2ecc71' if v >= 0.75 else '#e74c3c' for v in values]
    bars = ax.bar(metrics, [v*100 for v in values], color=colors, width=0.5)
    ax.set_ylim(0, 110)
    ax.axhline(y=75, color='orange', linestyle='--', linewidth=1, label='75% threshold')
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance Metrics')
    ax.legend(fontsize=8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    charts.append({"title": "Performance Metrics", "img": fig_to_base64(fig)})

    results.append({
        "check": "Accuracy",
        **pass_fail(test_acc >= 0.75,
                    f"Accuracy = {test_acc*100:.1f}% — Good!",
                    f"Accuracy = {test_acc*100:.1f}% — Too low!"),
        "detail": {"train": f"{train_acc*100:.1f}%", "test": f"{test_acc*100:.1f}%"}
    })

    results.append({
        "check": "Precision / Recall / F1",
        **pass_fail(f1 >= 0.75,
                    f"F1={f1:.3f} Precision={precision:.3f} Recall={recall:.3f}",
                    f"F1={f1:.3f} — too low!"),
        "detail": {"precision": round(precision,3), "recall": round(recall,3), "f1": round(f1,3)}
    })

    # 2. Overfitting chart
    gap = train_acc - test_acc
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(['Train Accuracy', 'Test Accuracy'],
           [train_acc*100, test_acc*100],
           color=['#3498db', '#2ecc71' if gap <= 0.15 else '#e74c3c'])
    ax.set_ylim(0, 110)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Overfitting Check (gap={gap*100:.1f}%)')
    for i, v in enumerate([train_acc*100, test_acc*100]):
        ax.text(i, v+1, f'{v:.1f}%', ha='center', fontweight='bold')
    plt.tight_layout()
    charts.append({"title": "Overfitting", "img": fig_to_base64(fig)})
    results.append({
        "check": "Overfitting",
        **pass_fail(gap <= 0.15,
                    f"No overfitting! Gap={gap*100:.1f}%",
                    f"Overfitting detected! Gap={gap*100:.1f}%"),
        "detail": {"gap": f"{gap*100:.1f}%"}
    })

    # 3. Confusion matrix
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    charts.append({"title": "Confusion Matrix", "img": fig_to_base64(fig)})
    results.append({
        "check": "Confusion Matrix",
        "status": "PASS", "msg": "Confusion matrix generated!",
        "detail": {"matrix": cm.tolist(), "classes": class_names}
    })

    # 4. Feature importance
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:10]
    fig, ax     = plt.subplots(figsize=(8, 4))
    ax.barh([feature_names[i] for i in reversed(indices)],
            [importances[i] for i in reversed(indices)],
            color=sns.color_palette("viridis", len(indices)))
    ax.set_title('Top 10 Feature Importances')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    charts.append({"title": "Feature Importance", "img": fig_to_base64(fig)})
    results.append({
        "check": "Explainability",
        "status": "PASS",
        "msg": "Model is explainable via feature importances!",
        "detail": {feature_names[i]: round(float(importances[i]),4) for i in indices[:5]}
    })

    # 5. Cross validation
    cv_scores = cross_val_score(
        RandomForestClassifier(random_state=42), X, y, cv=5)
    cv_std    = cv_scores.std()
    fig, ax   = plt.subplots(figsize=(6, 3))
    ax.bar(range(1, 6), cv_scores*100,
           color=['#2ecc71' if s >= 0.75 else '#e74c3c' for s in cv_scores])
    ax.axhline(y=cv_scores.mean()*100, color='blue', linestyle='--',
               label=f'Mean={cv_scores.mean()*100:.1f}%')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('5-Fold Cross Validation')
    ax.legend()
    plt.tight_layout()
    charts.append({"title": "Cross Validation", "img": fig_to_base64(fig)})
    results.append({
        "check": "Cross Validation",
        **pass_fail(cv_std <= 0.10 and cv_scores.mean() >= 0.75,
                    f"Stable! Mean={cv_scores.mean()*100:.1f}% Std={cv_std:.3f}",
                    f"Unstable! Std={cv_std:.3f} — model varies too much"),
        "detail": {"mean": f"{cv_scores.mean()*100:.1f}%", "std": round(float(cv_std),4)}
    })

    # 6. Per-class fairness
    report      = classification_report(y_test, y_pred,
                                        target_names=class_names, output_dict=True,
                                        zero_division=0)
    class_f1s   = {c: round(report[c]['f1-score'],3)
                   for c in class_names if c in report}
    fairness_ok = all(v >= 0.60 for v in class_f1s.values())
    fig, ax     = plt.subplots(figsize=(6, 3))
    ax.bar(list(class_f1s.keys()), list(class_f1s.values()),
           color=['#2ecc71' if v >= 0.60 else '#e74c3c'
                  for v in class_f1s.values()])
    ax.axhline(y=0.60, color='orange', linestyle='--', label='60% threshold')
    ax.set_title('Per-Class F1 — Fairness Check')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    charts.append({"title": "Fairness", "img": fig_to_base64(fig)})
    results.append({
        "check": "Per-Class Fairness",
        **pass_fail(fairness_ok,
                    "Model is fair across all classes!",
                    f"Unfair to some classes: {[c for c,v in class_f1s.items() if v<0.60]}"),
        "detail": class_f1s
    })

    # Summary
    passed  = sum(1 for r in results if r["status"] == "PASS")
    failed  = len(results) - passed
    if   failed == 0:    overall = "MODEL IS READY FOR DEPLOYMENT!"
    elif failed <= 2:    overall = "MODEL NEEDS MINOR FIXES."
    else:                overall = "SERIOUS MODEL ISSUES — DO NOT DEPLOY!"

    return {"results": results, "charts": charts,
            "passed": passed, "failed": failed,
            "overall": overall, "total": len(results),
            "train_acc": round(train_acc*100,1),
            "test_acc":  round(test_acc*100,1),
            "f1": round(f1,3)}

# ─────────────────────────────────────────────
# MODULE 3 — RISK ASSESSMENT
# ─────────────────────────────────────────────

RISK_CATEGORIES = {
    "Strategic":   [(1,"AI strategy not in sync with org values"),(2,"Lack of management vision"),(3,"Lack of model interdependency knowledge")],
    "Financial":   [(4,"Direct financial losses from incorrect models")],
    "Data":        [(5,"Test data different from production"),(6,"Inaccurate or biased data"),(7,"Unauthorized data access"),(8,"Lack of data sensitivity tagging")],
    "Technology":  [(9,"Black box — no auditability"),(10,"No monitoring mechanism"),(11,"Single point of failure"),(12,"Cannot scale")],
    "Algorithmic": [(13,"Biased training data"),(14,"No stress testing"),(15,"Black box AI logic"),(16,"Insecure coding"),(17,"Unapproved changes to production"),(18,"Flawed model design"),(19,"No model drift detection")],
    "Cyber":       [(20,"PII/PHI not secured"),(21,"No opt-out for data sharing"),(22,"Data used without consent"),(23,"No access controls")],
    "People":      [(24,"AI talent risk"),(25,"No skilled AI talent"),(26,"Undefined roles"),(27,"Unclear human-machine interaction"),(28,"Loss of expertise"),(29,"Non-diverse workforce")],
    "Regulatory":  [(30,"Unclear AI regulations"),(31,"No AI governance body"),(32,"No disaster recovery")],
    "External":    [(33,"Network availability issues"),(34,"Incomplete interfaces")],
    "Third Party": [(35,"Unclear vendor roles"),(36,"Ineffective vendor risk mgmt"),(37,"Vendor security gaps"),(38,"License issues")],
    "Societal":    [(39,"Ignoring societal impact"),(40,"Non-transparent AI"),(41,"Socio-economic inequality")]
}

SEVERITY_SCORES = {"Low": 0.1, "Medium": 0.5, "High": 0.9}
IMPACT_SCORES   = {"Low": 2, "Medium": 5, "High": 8, "Critical": 10}

def run_risk_assessment(selected_risks, risk_ratings):
    results = []
    charts  = []

    # Build scoring
    scores      = {}
    weighted    = {}
    for rt, rating in risk_ratings.items():
        if rating["likelihood"] and rating["impact"]:
            l = SEVERITY_SCORES.get(rating["likelihood"], 0)
            i = IMPACT_SCORES.get(rating["impact"], 0)
            w = round(l * i, 3)
            scores[rt]   = {"likelihood": rating["likelihood"],
                            "impact": rating["impact"], "score": w}
            weighted[rt] = w

    # Overall risk score (0-10)
    if weighted:
        max_possible  = 0.9 * 10
        total_w       = sum(weighted.values())
        overall_risk  = round((total_w / (len(weighted) * max_possible)) * 10, 2)
        overall_risk  = min(overall_risk, 10)
    else:
        overall_risk  = 0.5

    # Risk level
    if   overall_risk < 2: level, color = "LOW",          "#2ecc71"
    elif overall_risk < 4: level, color = "MODERATE",     "#f1c40f"
    elif overall_risk < 6: level, color = "CONSIDERABLE", "#e67e22"
    elif overall_risk < 8: level, color = "HIGH",         "#e74c3c"
    else:                  level, color = "VERY HIGH",    "#8e44ad"

    # Autonomy
    if   overall_risk < 2: autonomy = "Level 5 — Full Autonomy"
    elif overall_risk < 4: autonomy = "Level 4 — High Autonomy"
    elif overall_risk < 6: autonomy = "Level 3 — Partial Autonomy"
    elif overall_risk < 8: autonomy = "Level 2 — Assisted"
    else:                  autonomy = "Level 1 — Manual Control"

    # Chart 1 — Risk score gauge (bar)
    fig, ax = plt.subplots(figsize=(6, 3))
    bar_color = color
    ax.barh(['Risk Score'], [overall_risk], color=bar_color, height=0.4)
    ax.set_xlim(0, 10)
    ax.axvline(x=2, color='green',  linestyle='--', alpha=0.5, label='Low')
    ax.axvline(x=4, color='yellow', linestyle='--', alpha=0.5, label='Moderate')
    ax.axvline(x=6, color='orange', linestyle='--', alpha=0.5, label='Considerable')
    ax.axvline(x=8, color='red',    linestyle='--', alpha=0.5, label='High')
    ax.text(overall_risk + 0.1, 0, f'{overall_risk}/10', va='center', fontweight='bold', fontsize=12)
    ax.set_title(f'Overall Risk Score — {level}')
    ax.legend(fontsize=7, loc='lower right')
    plt.tight_layout()
    charts.append({"title": "Risk Score", "img": fig_to_base64(fig)})

    # Chart 2 — Risk type breakdown
    if weighted:
        fig, ax = plt.subplots(figsize=(7, 4))
        types   = list(weighted.keys())
        vals    = list(weighted.values())
        c_map   = ['#e74c3c' if v > 5 else '#f1c40f' if v > 2 else '#2ecc71' for v in vals]
        ax.bar(types, vals, color=c_map)
        ax.set_ylabel('Weighted Risk Score')
        ax.set_title('Risk Score by Risk Type')
        ax.axhline(y=3, color='orange', linestyle='--', alpha=0.6, label='Threshold')
        ax.legend()
        plt.tight_layout()
        charts.append({"title": "Risk Breakdown", "img": fig_to_base64(fig)})

    # Chart 3 — Applicable risks by category
    cat_counts = {}
    for num, cat, desc in selected_risks:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    if cat_counts:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(cat_counts.values(), labels=cat_counts.keys(),
               autopct='%1.0f%%', startangle=140,
               colors=sns.color_palette("Set3", len(cat_counts)))
        ax.set_title(f'Applicable Risks by Category ({len(selected_risks)} total)')
        plt.tight_layout()
        charts.append({"title": "Risk Categories", "img": fig_to_base64(fig)})

    # Chart 4 — Likelihood vs Impact matrix
    if scores:
        LIKE_NUM = {"Low": 1, "Medium": 2, "High": 3}
        IMP_NUM  = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        fig, ax  = plt.subplots(figsize=(6, 4))
        for rt, s in scores.items():
            x = LIKE_NUM.get(s["likelihood"], 1)
            y = IMP_NUM.get(s["impact"], 1)
            c = '#e74c3c' if s["score"] > 4 else '#f1c40f' if s["score"] > 1 else '#2ecc71'
            ax.scatter(x, y, s=200, color=c, zorder=5)
            ax.annotate(rt, (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=7)
        ax.set_xticks([1,2,3])
        ax.set_xticklabels(['Low','Medium','High'])
        ax.set_yticks([1,2,3,4])
        ax.set_yticklabels(['Low','Medium','High','Critical'])
        ax.set_xlabel('Likelihood')
        ax.set_ylabel('Impact')
        ax.set_title('Risk Matrix (Likelihood vs Impact)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        charts.append({"title": "Risk Matrix", "img": fig_to_base64(fig)})

    results.append({
        "check":  "Overall Risk Score",
        "status": "PASS" if overall_risk < 4 else "FAIL",
        "msg":    f"Score={overall_risk}/10 — {level}",
        "detail": {"score": overall_risk, "level": level}
    })
    results.append({
        "check":  "Autonomy Level",
        "status": "PASS",
        "msg":    autonomy,
        "detail": scores
    })
    results.append({
        "check":  "Applicable Risks",
        "status": "PASS" if len(selected_risks) < 10 else "FAIL",
        "msg":    f"{len(selected_risks)} risks identified across {len(cat_counts)} categories",
        "detail": cat_counts
    })

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = len(results) - passed

    return {"results": results, "charts": charts,
            "passed": passed, "failed": failed,
            "overall_risk": overall_risk, "level": level,
            "autonomy": autonomy, "total": len(results),
            "selected_risks": [(n, c, d) for n, c, d in selected_risks],
            "risk_categories": list(RISK_CATEGORIES.keys())}

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    risk_cats = {cat: risks for cat, risks in RISK_CATEGORIES.items()}
    return render_template('index.html', risk_categories=risk_cats)

@app.route('/validate_data', methods=['POST'])
def validate_data():
    try:
        file   = request.files['file']
        target = request.form['target_column'].strip()
        df     = pd.read_csv(file)
        if target not in df.columns:
            return jsonify({"error": f"Column '{target}' not found. Available: {list(df.columns)}"}), 400
        res = run_data_validation(df, target)
        res["columns"] = list(df.columns)
        res["shape"]   = list(df.shape)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/validate_model', methods=['POST'])
def validate_model():
    try:
        file   = request.files['file']
        target = request.form['target_column'].strip()
        df     = pd.read_csv(file)
        if target not in df.columns:
            return jsonify({"error": f"Column '{target}' not found."}), 400
        res = run_model_validation(df, target)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/assess_risk', methods=['POST'])
def assess_risk():
    try:
        data          = request.get_json()
        selected_nums = data.get('selected_risks', [])
        risk_ratings  = data.get('risk_ratings', {})

        all_risks     = [(n, cat, desc)
                         for cat, risks in RISK_CATEGORIES.items()
                         for n, desc in risks]
        selected      = [(n, c, d) for n, c, d in all_risks if n in selected_nums]
        res           = run_risk_assessment(selected, risk_ratings)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_columns', methods=['POST'])
def get_columns():
    try:
        file = request.files['file']
        df   = pd.read_csv(file)
        return jsonify({"columns": list(df.columns), "shape": list(df.shape)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)