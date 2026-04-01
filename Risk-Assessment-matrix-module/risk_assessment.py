import pandas as pd
import numpy as np
from itertools import groupby

# ============================================================
# AI V&V RISK ASSESSOR - EASY VERSION
# 
# CHANGES FROM OLD VERSION:
#   - 41 risks shown as a menu — just type numbers
#   - No manual probability entry — pick Low/Med/High
#   - System auto-normalizes probabilities to sum to 1.0
#   - Clean readable output with clear Pass/Fail
#   - Report saved automatically
# ============================================================

# ─────────────────────────────────────────────
# RISK DEFINITIONS
# ─────────────────────────────────────────────

RISK_TYPES = {
    "PD": "Performance Degradation",
    "FO": "Failure of Operation",
    "ID": "Irreversible Damage",
    "DE": "Destruction to Environment",
    "HC": "Human Casualties"
}

# Severity options — user picks 1/2/3 instead of typing decimals
SEVERITY_MAP = {
    "1": ("Low",    0.1),
    "2": ("Medium", 0.5),
    "3": ("High",   0.9)
}

# Impact options — user picks 1/2/3 instead of typing 0-10
IMPACT_MAP = {
    "1": ("Low",      2),
    "2": ("Medium",   5),
    "3": ("High",     8),
    "4": ("Critical", 10)
}

# ─────────────────────────────────────────────
# 41 RISKS GROUPED BY CATEGORY
# ─────────────────────────────────────────────

RISK_CATEGORIES = {
    "Strategic": [
        (1,  "AI strategy not in sync with organizational values"),
        (2,  "Lack of management vision on AI adoption"),
        (3,  "Lack of knowledge about model interdependencies"),
    ],
    "Financial": [
        (4,  "Direct financial losses from incorrect models"),
    ],
    "Data": [
        (5,  "Test data very different from production data"),
        (6,  "Non-availability of accurate or unbiased data"),
        (7,  "Unauthorized access to AI models or training data"),
        (8,  "Lack of data sensitivity tagging"),
    ],
    "Technology": [
        (9,  "Lack of auditability — black box effect"),
        (10, "Lack of monitoring mechanism for model output"),
        (11, "Lack of redundancy — single point of failure"),
        (12, "Rigidness of technology to scale"),
    ],
    "Algorithmic": [
        (13, "Biased data used to train AI models"),
        (14, "Lack of risk-based stress testing"),
        (15, "AI logic not transparent — black box"),
        (16, "Secure coding practices not maintained"),
        (17, "Un-approved changes moved to production"),
        (18, "Flaws in model design — no audit trail"),
        (19, "Lack of mechanisms to detect model drift"),
    ],
    "Cyber": [
        (20, "PII and PHI data not secured enough"),
        (21, "Customers cannot opt-out from data sharing"),
        (22, "Data used without explicit consent"),
        (23, "Lack of access controls for data/logic"),
    ],
    "People": [
        (24, "Risk to organizational talent due to AI"),
        (25, "Lack of skilled talent to build/train/deploy AI"),
        (26, "Lack of defined roles and responsibilities"),
        (27, "Lack of clarity on human-machine interaction"),
        (28, "Loss of organizational expertise"),
        (29, "Non-diverse workforce leading to biased models"),
    ],
    "Regulatory": [
        (30, "Lack of clarity on AI regulations"),
        (31, "No internal governance body for AI oversight"),
        (32, "No disaster recovery plans"),
    ],
    "External": [
        (33, "Non-availability of AI due to network issues"),
        (34, "Incomplete interfaces between AI solutions"),
    ],
    "Third Party": [
        (35, "Lack of clarity on vendor roles"),
        (36, "Ineffective risk management with vendors"),
        (37, "Lack of security protocols by third party"),
        (38, "Vendor license requirements not met"),
    ],
    "Societal": [
        (39, "Indifference to societal impact of AI"),
        (40, "Non-transparent AI excluding certain individuals"),
        (41, "Widening socio-economic inequality from AI"),
    ]
}

# ─────────────────────────────────────────────
# INFEASIBLE COMBINATIONS (from professor doc)
# ─────────────────────────────────────────────

INFEASIBLE = [
    (0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0),(0,0,0,0,1),
    (1,0,1,0,0),(1,0,0,1,0),(1,0,0,0,1),
]

# ─────────────────────────────────────────────
# AUTONOMY LEVEL MAPPING
# ─────────────────────────────────────────────

def get_autonomy_level(score):
    if score < 2:
        return "Level 5 — Full Autonomy",    "AI makes decisions alone (HOOL)", "Self-adapting test loops"
    elif score < 4:
        return "Level 4 — High Autonomy",    "Mostly independent (HOL)",         "Continuous dynamic testing"
    elif score < 6:
        return "Level 3 — Partial Autonomy", "Human helps on complex tasks (HOL)","Continuous static testing"
    elif score < 8:
        return "Level 2 — Assisted",         "Human oversight needed (HOL)",      "Basic ML test automation"
    else:
        return "Level 1 — Manual Control",   "Constant human control (HIL)",      "Manual testing + analytics"

# ─────────────────────────────────────────────
# HELPER — DIVIDER LINE
# ─────────────────────────────────────────────

def divider(char="─", width=55):
    print(char * width)

# ─────────────────────────────────────────────
# STEP 1 — SHOW RISKS AS MENU
# user just types numbers like: 5 6 13
# ─────────────────────────────────────────────

def step1_identify_risks():
    divider("=")
    print("  STEP 1 — WHICH RISKS APPLY TO YOUR AI?")
    divider("=")
    print("\nBelow are 41 risks grouped by category.")
    print("Just type the NUMBERS that apply to your AI.")
    print("Example: if risks 5, 6 and 13 apply → type: 5 6 13\n")

    # Print all risks grouped
    for category, risks in RISK_CATEGORIES.items():
        divider()
        print(f"  {category.upper()} RISKS")
        divider()
        for num, desc in risks:
            print(f"  [{num:02d}] {desc}")

    print()
    divider("=")

    # Get user input
    while True:
        raw = input("  Type applicable risk numbers (space separated): ").strip()
        if not raw:
            print("  Please enter at least one number!")
            continue
        try:
            chosen = list(map(int, raw.split()))
            # validate all numbers are 1-41
            valid_nums = [n for cat in RISK_CATEGORIES.values() for n,_ in cat]
            invalid = [n for n in chosen if n not in valid_nums]
            if invalid:
                print(f"  Invalid numbers: {invalid} — please use 1 to 41 only")
                continue
            break
        except ValueError:
            print("  Please type numbers only like: 5 6 13")

    # Build applicable list
    all_risks = [(n, cat, desc)
                 for cat, risks in RISK_CATEGORIES.items()
                 for n, desc in risks]
    applicable = [(n, cat, desc) for n, cat, desc in all_risks if n in chosen]

    print(f"\n  You selected {len(applicable)} risks.")
    print("  These will be used in your risk report.\n")
    return applicable

# ─────────────────────────────────────────────
# STEP 2 — RATE EACH RISK TYPE (simple menu)
# instead of 25 scenario questions
# ─────────────────────────────────────────────

def step2_rate_risk_types():
    divider("=")
    print("  STEP 2 — RATE EACH RISK TYPE")
    divider("=")
    print("""
Your AI can face 5 types of failures.
Rate each one:
  How LIKELY is it?   → 1=Low  2=Medium  3=High
  How BAD would it be? → 1=Low  2=Medium  3=High  4=Critical

Just press ENTER to skip (means Not Applicable).
""")

    risk_ratings = {}

    for code, name in RISK_TYPES.items():
        divider()
        print(f"\n  {code} — {name}")

        # Likelihood
        while True:
            like = input(f"  Likelihood  (1=Low 2=Med 3=High, Enter=skip): ").strip()
            if like == "":
                risk_ratings[code] = None
                break
            if like in SEVERITY_MAP:
                # Impact
                while True:
                    imp = input(f"  Impact      (1=Low 2=Med 3=High 4=Critical):  ").strip()
                    if imp in IMPACT_MAP:
                        like_label, like_val = SEVERITY_MAP[like]
                        imp_label,  imp_val  = IMPACT_MAP[imp]
                        risk_ratings[code] = {
                            "likelihood_label": like_label,
                            "likelihood_val":   like_val,
                            "impact_label":     imp_label,
                            "impact_val":       imp_val,
                            "weighted":         round(like_val * imp_val, 4)
                        }
                        print(f"  Saved: {like_label} likelihood × {imp_label} impact = {like_val * imp_val:.2f}")
                        break
                    else:
                        print("  Please type 1, 2, 3 or 4")
                break
            else:
                print("  Please type 1, 2 or 3 (or Enter to skip)")

    return risk_ratings

# ─────────────────────────────────────────────
# STEP 3 — CALCULATE SCORE + SHOW RESULT
# ─────────────────────────────────────────────

def step3_calculate_and_report(applicable_risks, risk_ratings):
    divider("=")
    print("  STEP 3 — RESULTS")
    divider("=")

    # Calculate overall risk score (0-10)
    active = {k: v for k, v in risk_ratings.items() if v is not None}

    if not active:
        print("\n  No risk types rated — defaulting to low risk score.")
        overall_risk = 0.5
    else:
        # Normalize weighted impacts to 0-10 scale
        max_possible = 0.9 * 10  # max likelihood × max impact
        total_weighted = sum(v["weighted"] for v in active.values())
        overall_risk = round((total_weighted / (len(active) * max_possible)) * 10, 2)
        overall_risk = min(overall_risk, 10)  # cap at 10

    autonomy_level, autonomy_desc, testing_level = get_autonomy_level(overall_risk)

    # Risk level label
    if overall_risk < 2:
        risk_label = "LOW"
        risk_emoji = "GREEN"
    elif overall_risk < 4:
        risk_label = "MODERATE"
        risk_emoji = "YELLOW"
    elif overall_risk < 6:
        risk_label = "CONSIDERABLE"
        risk_emoji = "ORANGE"
    elif overall_risk < 8:
        risk_label = "HIGH"
        risk_emoji = "RED"
    else:
        risk_label = "VERY HIGH"
        risk_emoji = "CRITICAL"

    # Print results
    print(f"""
  Overall Risk Score : {overall_risk:.2f} / 10
  Risk Level         : {risk_label} ({risk_emoji})
  Applicable Risks   : {len(applicable_risks)} identified

  Autonomy Level     : {autonomy_level}
  Description        : {autonomy_desc}
  Testing Required   : {testing_level}
""")

    divider()
    print("  RISK TYPE BREAKDOWN")
    divider()
    print(f"  {'Risk Type':<30} {'Likelihood':<12} {'Impact':<12} {'Score'}")
    divider()

    for code, name in RISK_TYPES.items():
        rating = risk_ratings.get(code)
        if rating:
            print(f"  {name:<30} {rating['likelihood_label']:<12} {rating['impact_label']:<12} {rating['weighted']:.2f}")
        else:
            print(f"  {name:<30} {'N/A':<12} {'N/A':<12} 0.00")

    divider()
    print(f"\n  What this means:")
    if overall_risk < 2:
        print("  Your AI is LOW risk.")
        print("  Can operate with full autonomy.")
        print("  Minimal human oversight needed.")
    elif overall_risk < 4:
        print("  Your AI is MODERATE risk.")
        print("  Mostly independent operation OK.")
        print("  Some human review recommended.")
    elif overall_risk < 6:
        print("  Your AI has CONSIDERABLE risk.")
        print("  Human intervention needed for complex decisions.")
        print("  Regular monitoring required.")
    elif overall_risk < 8:
        print("  Your AI is HIGH risk.")
        print("  Human oversight mandatory.")
        print("  Strict testing and monitoring required.")
    else:
        print("  Your AI is VERY HIGH risk.")
        print("  Constant human control required.")
        print("  Follows EU AI Act high-risk category rules.")
        print("  External audit may be required.")

    # Applicable risks grouped
    if applicable_risks:
        print()
        divider()
        print("  APPLICABLE RISKS BY CATEGORY")
        divider()
        applicable_risks.sort(key=lambda x: x[1])
        for category, group in groupby(applicable_risks, key=lambda x: x[1]):
            items = list(group)
            print(f"\n  {category} ({len(items)} risks):")
            for num, cat, desc in items:
                print(f"    [{num:02d}] {desc}")

    # Save report
    save_report(applicable_risks, risk_ratings, overall_risk,
                risk_label, autonomy_level, autonomy_desc, testing_level)

    return overall_risk

# ─────────────────────────────────────────────
# SAVE REPORT TO FILE
# ─────────────────────────────────────────────

def save_report(applicable_risks, risk_ratings, overall_risk,
                risk_label, autonomy_level, autonomy_desc, testing_level):

    lines = [
        "=" * 55,
        "  AI V&V RISK ASSESSMENT REPORT",
        "=" * 55,
        f"  Overall Risk Score : {overall_risk:.2f} / 10",
        f"  Risk Level         : {risk_label}",
        f"  Autonomy Level     : {autonomy_level}",
        f"  Description        : {autonomy_desc}",
        f"  Testing Required   : {testing_level}",
        f"  Applicable Risks   : {len(applicable_risks)}",
        "",
        "RISK TYPE BREAKDOWN:",
    ]

    for code, name in RISK_TYPES.items():
        r = risk_ratings.get(code)
        if r:
            lines.append(f"  {name}: {r['likelihood_label']} likelihood × "
                         f"{r['impact_label']} impact = {r['weighted']:.2f}")
        else:
            lines.append(f"  {name}: N/A")

    lines.append("\nAPPLICABLE RISKS:")
    for num, cat, desc in applicable_risks:
        lines.append(f"  [{num:02d}] {cat}: {desc}")

    with open("rpia_report.txt", "w") as f:
        f.write("\n".join(lines))

    # Save as CSV too
    rows = []
    for code, name in RISK_TYPES.items():
        r = risk_ratings.get(code)
        rows.append({
            "Risk Type": name,
            "Likelihood": r["likelihood_label"] if r else "N/A",
            "Impact":     r["impact_label"]     if r else "N/A",
            "Score":      r["weighted"]          if r else 0
        })
    pd.DataFrame(rows).to_csv("rpi_table.csv", index=False)

    print("\n  Report saved to : rpia_report.txt")
    print("  Table  saved to : rpi_table.csv")

# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run_risk_assessment():
    divider("=")
    print("  AI V&V RISK ASSESSOR — EASY VERSION")
    divider("=")
    print("""
  This tool does 3 simple steps:

  Step 1 — Pick which risks apply to your AI
           (just type numbers from the list)

  Step 2 — Rate each risk type
           (just pick Low / Medium / High)

  Step 3 — Get your risk score and report
           (saved automatically to file)

  Takes about 3 minutes total!
""")
    input("  Press Enter to start...")

    applicable = step1_identify_risks()
    ratings    = step2_rate_risk_types()
    score      = step3_calculate_and_report(applicable, ratings)

    divider("=")
    print("  ASSESSMENT COMPLETE!")
    divider("=")

# --- RUN ---
run_risk_assessment()