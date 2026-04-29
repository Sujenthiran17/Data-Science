# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import pandas as pd
import numpy as np

spark = SparkSession.builder.getOrCreate()

# ── Load ──────────────────────────────────────────────────────────────────────
df_spark = spark.table("`workspace`.`ds-raw-datasets`.`joined_two_tables_removed_nulls`")
df = df_spark.toPandas()

target        = "prospect_outcome"
target_binary = (df[target] == "Churned").astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# HYPOTHESIS FRAMEWORK
# ─────────────────────────────────────────────────────────────────────────────
#
# CHI-SQUARE TEST (Categorical features)
# ───────────────────────────────────────
# H0: The distribution of [feature] is INDEPENDENT of prospect_outcome
#     (i.e. the feature has NO association with churn vs won)
# H1: The distribution of [feature] is DEPENDENT on prospect_outcome
#     (i.e. the feature IS associated with churn vs won)
# Decision: Reject H0 if p < bonferroni_alpha → feature is significant
# Effect size: Cramér's V (0=no association, 1=perfect association)
#
# T-TEST / MANN-WHITNEY U (Continuous features)
# ───────────────────────────────────────────────
# H0: mean([feature] | Won) == mean([feature] | Churned)
#     (i.e. the feature does NOT differ between churned and won customers)
# H1: mean([feature] | Won) != mean([feature] | Churned)
#     (i.e. the feature DOES differ between churned and won customers)
# Decision: Reject H0 if p < bonferroni_alpha → feature is significant
# Effect size: Cohen's d (0.2=small, 0.5=medium, 0.8=large)
#
# Multiple testing correction: Bonferroni (α = 0.05 / number of tests)
# This controls the family-wise error rate — reduces false positives
# from running many simultaneous hypothesis tests.
# ─────────────────────────────────────────────────────────────────────────────

DROP_COLS = {
    "co_ref", "prospect_outcome", "renewal_month", "date_time_out",
    "proforma_date", "registration_date", "prospect_renewal_date",
    "closed_date", "call_date", "last_years_date_paid", "last_renewal",
    "current_anchor_list", "competitor_benefits_mentioned",
    "proforma_approved_lists", "call_id", "index",
    "price_range_mentioned", "desire_to_cancel", "analysed_call",
    "c20", "competitor_value_comparison", "has_world_pay_token",
}

CHI_SQ_COLS = [
    "current_auto_renewal_flag", "current_world_pay_token",
    "proforma_auto_renewal", "proforma_world_pay_token",
    "mentioned_competitors", "membership_renewal_decision",
    "band", "last_band", "renewal_year", "payment_method",
    "prospect_status", "proforma_account_stage", "proforma_audit_status",
    "proforma_membership_status", "churn_category", "complaint_category",
    "customer_reaction_category", "agent_renewal_pitch_category",
    "customer_renewal_response_category", "agent_response_category",
    "topic_introduced_by", "customer_response", "call_direction",
    "justification_category", "reason_for_renewal_category",
    "agent_response_to_cancel_category",
    "argument_that_convinced_customer_to_stay_category",
    "serious_complaint", "other_complaint", "discussion_on_price_increase",
    "renewal_impact_due_to_price_increase", "discount_or_waiver_requested",
    "call_reschedule_request", "agent_flagged_membership_status_alert",
    "agent_renewal_initiation", "explicit_competitor_mention",
    "explicit_switching_intent", "price_switching_mentioned",
    "percentage_price_increase_mentioned", "monetary_price_increase_mentioned",
    "customer_asked_for_justification", "discount_offered",
    "proforma_date_is_null", "registration_date_is_null",
    "call_year", "call_number",
]

T_TEST_COLS = [
    "sustainability_score", "total_renewal_score_new", "auto_renewal_score",
    "status_scores", "anchoring_score", "tenure_scores", "current_anchorings",
    "renewal_score_at_release", "starting_net", "starting_vat",
    "starting_gross", "starting_membership_net", "starting_package_net",
    "starting_pqq_net", "gross", "membership_net", "package_net", "pqq_net",
    "amount", "total_amount", "datediff", "connection_net", "connection_qty",
    "starting_connection_net", "starting_connection_qty", "of_connection",
    "last_connections", "last_years_price", "total_net_paid",
    "last_total_net_paid", "tenure_years", "discount_amount", "payment_timeframe",
]

# ─────────────────────────────────────────────────────────────────────────────
# ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def encode_binary_str(series):
    mapping = {"y": 1, "n": 0, "true": 1, "false": 0,
               "yes": 1, "no": 0, "unknown": np.nan}
    return series.str.lower().map(mapping)

def standardise_call_direction(series):
    return series.str.upper().str.replace("_", "", regex=False)\
                 .map({"OUTBOUND": "Outbound", "INBOUND": "Inbound"})

df["current_auto_renewal_flag"]    = encode_binary_str(df["current_auto_renewal_flag"])
df["current_world_pay_token"]      = encode_binary_str(df["current_world_pay_token"])
df["proforma_auto_renewal"]        = encode_binary_str(df["proforma_auto_renewal"])
df["proforma_world_pay_token"]     = encode_binary_str(df["proforma_world_pay_token"])
df["mentioned_competitors"]        = encode_binary_str(df["mentioned_competitors"])
df["membership_renewal_decision"]  = encode_binary_str(df["membership_renewal_decision"])
df["call_direction"]               = standardise_call_direction(df["call_direction"])
df["tenure_group"]                 = df["tenure_group"].replace("4+", "4").astype(float)

T_TEST_COLS.append("tenure_group")
if "tenure_group" in CHI_SQ_COLS:
    CHI_SQ_COLS.remove("tenure_group")

ordinal_map = {"1": 1, "2": 2, "3": 3, "4 to 9": 6.5, "10+": 10}
df["anchor_group"]     = df["anchor_group"].map(ordinal_map)
df["connection_group"] = df["connection_group"].map(ordinal_map)
T_TEST_COLS += ["anchor_group", "connection_group"]
for c in ["anchor_group", "connection_group"]:
    if c in CHI_SQ_COLS:
        CHI_SQ_COLS.remove(c)

# ─────────────────────────────────────────────────────────────────────────────
# BONFERRONI CORRECTION
# ─────────────────────────────────────────────────────────────────────────────
total_tests      = len(CHI_SQ_COLS) + len(T_TEST_COLS)
bonferroni_alpha = 0.05 / total_tests

print("=" * 65)
print("HYPOTHESIS TESTING FOR CHURN PREDICTION")
print("=" * 65)
print(f"\nTarget variable  : {target}  (Won=0, Churned=1)")
print(f"Total tests      : {total_tests}  ({len(CHI_SQ_COLS)} chi-square + {len(T_TEST_COLS)} t-test)")
print(f"Raw α            : 0.05")
print(f"Bonferroni α     : 0.05 / {total_tests} = {bonferroni_alpha:.6f}")
print(f"\nCHI-SQUARE HYPOTHESES (per categorical feature):")
print(f"  H0: [feature] is independent of {target} — no association")
print(f"  H1: [feature] is dependent on {target}  — significant association")
print(f"\nT-TEST HYPOTHESES (per continuous feature):")
print(f"  H0: mean([feature] | Won) = mean([feature] | Churned)")
print(f"  H1: mean([feature] | Won) ≠ mean([feature] | Churned)")
print(f"\nDecision rule: Reject H0 if p_value < {bonferroni_alpha:.6f}")

won_mask   = df[target] == "Won"
churn_mask = df[target] == "Churned"

# ─────────────────────────────────────────────────────────────────────────────
# 1. CHI-SQUARE TEST
# ─────────────────────────────────────────────────────────────────────────────
chi_results = []

for col in CHI_SQ_COLS:
    if col not in df.columns:
        continue
    try:
        temp = df[[col, target]].dropna()
        ct   = pd.crosstab(temp[col], temp[target])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            raise ValueError("Degenerate contingency table — only one category present")

        chi2, p, dof, expected = chi2_contingency(ct)

        # Cramér's V effect size
        n = ct.values.sum()
        v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

        # Effect size interpretation
        if   v >= 0.5: effect_label = "Very Strong"
        elif v >= 0.3: effect_label = "Strong"
        elif v >= 0.1: effect_label = "Moderate"
        else:          effect_label = "Weak"

        reject_h0 = p < bonferroni_alpha

        chi_results.append({
            "Feature"         : col,
            "H0"              : f"'{col}' is independent of {target}",
            "H1"              : f"'{col}' is associated with {target}",
            "Chi2_Statistic"  : round(chi2, 4),
            "Degrees_Freedom" : dof,
            "P_Value"         : round(p, 8),
            "CramersV"        : round(v, 4),
            "Effect_Strength" : effect_label,
            "Reject_H0"       : "YES — significant" if reject_h0 else "NO  — not significant",
            "Conclusion"      : (f"'{col}' IS associated with churn (use in model)"
                                 if reject_h0 else
                                 f"'{col}' is NOT associated with churn (consider dropping)"),
        })
    except Exception as e:
        chi_results.append({
            "Feature": col, "H0": None, "H1": None,
            "Chi2_Statistic": None, "Degrees_Freedom": None,
            "P_Value": None, "CramersV": None,
            "Effect_Strength": None,
            "Reject_H0": f"ERROR", "Conclusion": str(e),
        })

# ─────────────────────────────────────────────────────────────────────────────
# 2. T-TEST + MANN-WHITNEY U
# ─────────────────────────────────────────────────────────────────────────────
t_results = []

for col in T_TEST_COLS:
    if col not in df.columns:
        continue
    try:
        won_vals   = df.loc[won_mask,   col].dropna()
        churn_vals = df.loc[churn_mask, col].dropna()

        if len(won_vals) < 2 or len(churn_vals) < 2:
            raise ValueError("Insufficient data in one group")

        # Welch's t-test (does not assume equal variance)
        t_stat, t_p = ttest_ind(won_vals, churn_vals, equal_var=False)

        # Mann-Whitney U (non-parametric — no normality assumption)
        u_stat, u_p = mannwhitneyu(won_vals, churn_vals, alternative="two-sided")

        # Cohen's d
        pooled_std = np.sqrt((won_vals.std()**2 + churn_vals.std()**2) / 2)
        cohens_d   = abs(won_vals.mean() - churn_vals.mean()) / pooled_std if pooled_std > 0 else 0

        # Effect size interpretation
        if   cohens_d >= 0.8: effect_label = "Large"
        elif cohens_d >= 0.5: effect_label = "Medium"
        elif cohens_d >= 0.2: effect_label = "Small"
        else:                 effect_label = "Negligible"

        # Direction of effect
        direction = "higher in Churned" if churn_vals.mean() > won_vals.mean() else "higher in Won"

        reject_h0_t = t_p < bonferroni_alpha
        reject_h0_u = u_p < bonferroni_alpha
        reject_h0   = reject_h0_t or reject_h0_u

        t_results.append({
            "Feature"           : col,
            "H0"                : f"mean({col} | Won) = mean({col} | Churned)",
            "H1"                : f"mean({col} | Won) ≠ mean({col} | Churned)",
            "Won_Mean"          : round(float(won_vals.mean()),   4),
            "Churned_Mean"      : round(float(churn_vals.mean()), 4),
            "Direction"         : direction,
            "T_Statistic"       : round(float(t_stat), 4),
            "T_P_Value"         : round(float(t_p),    8),
            "U_Statistic"       : round(float(u_stat), 4),
            "U_P_Value"         : round(float(u_p),    8),
            "Cohens_d"          : round(float(cohens_d), 4),
            "Effect_Strength"   : effect_label,
            "Reject_H0 (t)"     : "YES" if reject_h0_t else "NO",
            "Reject_H0 (U)"     : "YES" if reject_h0_u else "NO",
            "Conclusion"        : (f"'{col}' differs significantly — {direction} (use in model)"
                                   if reject_h0 else
                                   f"'{col}' does NOT differ between groups (consider dropping)"),
        })
    except Exception as e:
        t_results.append({
            "Feature": col, "H0": None, "H1": None,
            "Won_Mean": None, "Churned_Mean": None, "Direction": None,
            "T_Statistic": None, "T_P_Value": None,
            "U_Statistic": None, "U_P_Value": None,
            "Cohens_d": None, "Effect_Strength": None,
            "Reject_H0 (t)": "ERROR", "Reject_H0 (U)": "ERROR",
            "Conclusion": str(e),
        })

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────────────────────────────────────
chi_df = pd.DataFrame(chi_results).sort_values("CramersV", ascending=False)
t_df   = pd.DataFrame(t_results).sort_values("Cohens_d",  ascending=False)

def to_spark_safe(pdf):
    return spark.createDataFrame(pdf.astype(str).fillna("N/A"))

print(f"\n{'='*65}")
print("CHI-SQUARE RESULTS  (sorted by Cramér's V — effect size)")
print(f"{'='*65}")
display(to_spark_safe(chi_df))

print(f"\n{'='*65}")
print("T-TEST + MANN-WHITNEY U RESULTS  (sorted by Cohen's d)")
print(f"{'='*65}")
display(to_spark_safe(t_df))

# ── Summary ───────────────────────────────────────────────────────────────────
chi_sig  = chi_df[chi_df["Reject_H0"].str.startswith("YES")]
t_sig    = t_df[t_df["Reject_H0 (t)"].eq("YES") | t_df["Reject_H0 (U)"].eq("YES")]
chi_drop = chi_df[chi_df["Reject_H0"].str.startswith("NO")]
t_drop   = t_df[t_df["Reject_H0 (t)"].eq("NO")  & t_df["Reject_H0 (U)"].eq("NO")]

print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
print(f"\n  H0 REJECTED  (keep for model) : {len(chi_sig)} categorical + {len(t_sig)} continuous")
print(f"  H0 RETAINED  (consider drop)  : {len(chi_drop)} categorical + {len(t_drop)} continuous")

print(f"\n  Significant categorical features (H0 rejected):")
for _, row in chi_sig[["Feature","CramersV","Effect_Strength"]].iterrows():
    print(f"    ✓  {row['Feature']:<55} V={row['CramersV']}  [{row['Effect_Strength']}]")

print(f"\n  Significant continuous features (H0 rejected):")
for _, row in t_sig[["Feature","Won_Mean","Churned_Mean","Cohens_d","Effect_Strength","Direction"]].iterrows():
    print(f"    ✓  {row['Feature']:<40} d={row['Cohens_d']}  [{row['Effect_Strength']}]  {row['Direction']}")

print(f"\n  Features where H0 was NOT rejected (consider dropping):")
all_drop = list(chi_drop["Feature"]) + list(t_drop["Feature"])
print(f"    {', '.join(all_drop)}")

# COMMAND ----------

df.describe()


# COMMAND ----------

df.select_dtypes(include=["number"]).corr()

# COMMAND ----------

df.select_dtypes(include=["object"]).describe().transpose()

# COMMAND ----------

df["competitor_benefits_mentioned"].unique()

# COMMAND ----------

from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score,
                             ConfusionMatrixDisplay)
from imblearn.over_sampling  import SMOTE
from imblearn.combine        import SMOTETomek
from imblearn.pipeline       import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder.getOrCreate()

# ── Use feature data from Cell 1 ─────────────────────────────────────────────
df = df.copy()
target = "prospect_outcome"

# ── Fix tenure_group BEFORE anything else ────────────────────────────────────
df["tenure_group"] = df["tenure_group"].replace("4+", "4").astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# SELECTED FEATURES (from hypothesis testing — H0 rejected, leakage removed)
# ─────────────────────────────────────────────────────────────────────────────

# Leakage columns — filled AFTER outcome is known — NEVER use these
LEAKAGE = {
    "prospect_status", "membership_renewal_decision", "churn_category",
    "agent_response_category", "customer_renewal_response_category",
    "agent_response_to_cancel_category", "customer_response",
    "complaint_category", "customer_reaction_category",
    "agent_renewal_pitch_category", "reason_for_renewal_category",
    "justification_category", "argument_that_convinced_customer_to_stay_category",
    "topic_introduced_by", "renewal_impact_due_to_price_increase",
    "discount_offered", "payment_method",
}

# Continuous features — scale before model
CONTINUOUS = [
    "total_renewal_score_new",
    "status_scores",
    "sustainability_score",
    "auto_renewal_score",
    "tenure_scores",
    "renewal_score_at_release",
    "anchoring_score",
    "datediff",
    "discount_amount",
    "tenure_years",
    "tenure_group",       # after replace("4+","4").astype(float)
    "of_connection",      # ← covers anchor_group + connection_group signal
    "gross",
    "total_amount",
    "last_years_price",
    "membership_net",
    "starting_pqq_net",
]

# Categorical features — encode before model
CATEGORICAL = [
    "current_world_pay_token",              # V=0.29 — binary
    "proforma_audit_status",                # V=0.24
    "proforma_account_stage",              # V=0.22
    "proforma_membership_status",          # V=0.20
    "current_auto_renewal_flag",           # V=0.17 — binary
    "call_direction",                      # V=0.15
    "mentioned_competitors",               # V=0.14 — binary
    "agent_flagged_membership_status_alert", # V=0.13 — binary
    "price_switching_mentioned",           # V=0.10 — binary
    "band",                                # V=0.09
    "last_band",                           # V=0.09
    "proforma_auto_renewal",               # V=0.05 — binary
    "explicit_switching_intent",           # V=0.04 — binary
    "serious_complaint",                   # V=0.04 — binary
    "explicit_competitor_mention",         # V=0.04 — binary
    "monetary_price_increase_mentioned",   # V=0.04 — binary
    "customer_asked_for_justification",    # V=0.04 — binary
]

ALL_FEATURES = CONTINUOUS + CATEGORICAL

# ── Validate all expected features exist before proceeding ───────────────────
missing = [c for c in ALL_FEATURES if c not in df.columns]
if missing:
    print(f"⚠️  WARNING: {len(missing)} features missing from table — dropping them")
    print(f"   Missing: {missing}")
    CONTINUOUS = [c for c in CONTINUOUS if c not in missing]
    CATEGORICAL = [c for c in CATEGORICAL if c not in missing]
    ALL_FEATURES = CONTINUOUS + CATEGORICAL

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

df_model = df[ALL_FEATURES + [target]].copy()

# Encode target
df_model[target] = (df_model[target] == "Churned").astype(int)   # 1=Churned, 0=Won

# Encode binary string columns → 0/1 (skip if already numeric)
def encode_binary(series):
    mapping = {"y": 1, "n": 0, "true": 1, "false": 0,
               "yes": 1, "no": 0, "unknown": np.nan}
    return series.astype(str).str.lower().map(mapping).astype(float)

binary_cols = [
    "current_world_pay_token", "current_auto_renewal_flag",
    "mentioned_competitors", "proforma_auto_renewal",
    "agent_flagged_membership_status_alert", "price_switching_mentioned",
    "explicit_switching_intent", "serious_complaint",
    "explicit_competitor_mention", "monetary_price_increase_mentioned",
    "customer_asked_for_justification",
]
for col in binary_cols:
    if col in df_model.columns:
        if df_model[col].dtype not in ['int32', 'int64', 'float32', 'float64']:
            df_model[col] = encode_binary(df_model[col])

# Label encode remaining nominal categoricals
nominal_cols = [c for c in CATEGORICAL if c not in binary_cols]
le = LabelEncoder()
for col in nominal_cols:
    if col in df_model.columns:
        df_model[col] = le.fit_transform(df_model[col].astype(str))

# Drop remaining NaNs (minimal given prior hypothesis test filtering)
df_model = df_model.dropna()

print(f"Final model dataset shape: {df_model.shape}")
print(f"Features: {len(ALL_FEATURES)} ({len(CONTINUOUS)} continuous + {len(CATEGORICAL)} categorical)")
print(f"Target distribution:\n{df_model[target].value_counts()}")

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN-TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
X = df_model[ALL_FEATURES]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set : {X_test.shape[0]} samples")
print(f"Class balance (train): {y_train.value_counts(normalize=True).to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# BASELINE MODELS (NO RESAMPLING)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("BASELINE MODELS (imbalanced data)")
print("="*65)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

baseline_results = []

for name, clf in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    
    baseline_results.append({
        "Model": name,
        "ROC_AUC": round(roc_auc, 4),
        "Avg_Precision": round(avg_prec, 4),
    })
    
    print(f"\n{name}:")
    print(f"  ROC-AUC: {roc_auc:.4f}  |  Avg Precision: {avg_prec:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Won", "Churned"]))

baseline_df = pd.DataFrame(baseline_results)
display(baseline_df)

# ─────────────────────────────────────────────────────────────────────────────
# SMOTE RESAMPLING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("MODELS WITH SMOTE RESAMPLING")
print("="*65)

smote_results = []

for name, clf in models.items():
    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("classifier", clf)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    
    smote_results.append({
        "Model": name,
        "ROC_AUC": round(roc_auc, 4),
        "Avg_Precision": round(avg_prec, 4),
    })
    
    print(f"\n{name} + SMOTE:")
    print(f"  ROC-AUC: {roc_auc:.4f}  |  Avg Precision: {avg_prec:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Won", "Churned"]))

smote_df = pd.DataFrame(smote_results)
display(smote_df)

# ─────────────────────────────────────────────────────────────────────────────
# BEST MODEL SELECTION + CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("CROSS-VALIDATION (5-fold stratified)")
print("="*65)

best_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("classifier", best_model)
])

cv_scores = cross_validate(
    pipeline, X_train, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=["roc_auc", "average_precision"],
    return_train_score=True
)

print(f"\nROC-AUC (mean ± std): {cv_scores['test_roc_auc'].mean():.4f} ± {cv_scores['test_roc_auc'].std():.4f}")
print(f"Avg Precision (mean ± std): {cv_scores['test_average_precision'].mean():.4f} ± {cv_scores['test_average_precision'].std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("FINAL MODEL EVALUATION (test set)")
print("="*65)

pipeline.fit(X_train, y_train)
y_pred_final = pipeline.predict(X_test)
y_proba_final = pipeline.predict_proba(X_test)[:, 1]

roc_auc_final = roc_auc_score(y_test, y_proba_final)
avg_prec_final = average_precision_score(y_test, y_proba_final)

print(f"\nROC-AUC: {roc_auc_final:.4f}")
print(f"Avg Precision: {avg_prec_final:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=["Won", "Churned"]))

cm = confusion_matrix(y_test, y_pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Won", "Churned"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — Final Model")
plt.show()

print("\n✓ Model training complete")

# COMMAND ----------

# Accuracy for training and testing data to check overfitting
train_accuracy = pipeline.score(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy : {test_accuracy:.4f}")

# COMMAND ----------

df.select_dtypes(include="object").columns

# COMMAND ----------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# ==============================
# LOAD DATA
# ==============================
df = df.copy()
target = "prospect_outcome"

# Fix tenure_group
if "tenure_group" in df.columns:
    df["tenure_group"] = df["tenure_group"].replace("4+", "4").astype(float)

# ==============================
# USE ALL COLUMNS (EXCEPT TARGET)
# ==============================
FEATURES = [col for col in df.columns if col != target]

df_model = df[FEATURES + [target]].copy()

# ==============================
# TARGET ENCODING
# ==============================
df_model[target] = (df_model[target] == "Churned").astype(int)

# ==============================
# ENCODE CATEGORICAL FEATURES
# ==============================
le = LabelEncoder()

for col in df_model.select_dtypes(include="object").columns:
    if col != target:
        df_model[col] = le.fit_transform(df_model[col].astype(str))

# ==============================
# HANDLE MISSING VALUES
# ==============================
df_model = df_model.dropna()

# ==============================
# TRAIN-TEST SPLIT
# ==============================
X = df_model.drop(columns=[target])
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# MODEL PIPELINE
# ==============================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# ==============================
# TRAIN MODEL
# ==============================
pipeline.fit(X_train, y_train)

# ==============================
# PREDICTION
# ==============================
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# ==============================
# EVALUATION
# ==============================
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# COMMAND ----------

df_model.display()

# COMMAND ----------

df.columns

# COMMAND ----------

len(df_model.columns)

# COMMAND ----------

