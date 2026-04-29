# Databricks notebook source
from pyspark.sql.functions import DataFrame

# COMMAND ----------

df_billings: DataFrame = spark.table("`ds-raw-datasets`.`raw_billings`")
df_billings.describe().display()

# COMMAND ----------

df_renuals = spark.table("workspace.`ds-raw-datasets`.raw_renewal_calls").toPandas()
df_renuals.columns

# COMMAND ----------

# Databricks / PySpark — Billings Cleaning + Correlation Matrix
# ---------------------------------------------------------------
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load + build labelled training set ─────────────────────────────────
df_raw: DataFrame = spark.table("`ds-raw-datasets`.`raw_billings`")
print(f"Raw shape: ({df_raw.count()}, {len(df_raw.columns)})")

# Keep only rows closed within 28 days of renewal date (your churn window)
# and add binary target: Won=0 (retained), Churned=1 (churned)
df: DataFrame = (
    df_raw
    .filter(F.col("prospect_outcome") != "Open")
    .withColumn("datediff", F.datediff("closed_date", "prospect_renewal_date"))
    .filter((F.col("datediff") >= 0) & (F.col("datediff") < 29))
    .withColumn(
        "churned",
        F.when(F.col("prospect_outcome") == "Won", 0.0).otherwise(1.0)
    )
    .drop("prospect_outcome", "datediff")   # drop — replaced by churned
)
print(f"Labelled training rows: {df.count()}")
print(f"Churn rate: {df.filter(F.col('churned') == 1).count() / df.count():.1%}")

# ── 2. Drop columns where >60% of values are null ─────────────────────────
NULL_THRESHOLD = 0.60
total_rows = df.count()

null_fracs = df.select([
    (F.sum(F.col(c).isNull().cast("int")) / total_rows).alias(c)
    for c in df.columns
]).collect()[0].asDict()

cols_to_drop = [c for c, frac in null_fracs.items() if frac > NULL_THRESHOLD]
print(f"Dropping {len(cols_to_drop)} high-null columns: {cols_to_drop}")
df = df.drop(*cols_to_drop)

# ── 3. Fix boolean columns stored as mixed str/bool ───────────────────────
bool_cols = ["Proforma_Auto_Renewal", "Proforma_World_Pay_Token"]
bool_cols = [c for c in bool_cols if c in df.columns]
for col in bool_cols:
    df = df.withColumn(col,
        F.when(F.col(col).cast("string").isin("true", "True", "1"), 1.0)
         .when(F.col(col).cast("string").isin("false", "False", "0"), 0.0)
         .otherwise(None).cast(DoubleType())
    )

# ── 4. Fix y/n flag columns ────────────────────────────────────────────────
yn_cols = ["Current_Auto_Renewal_Flag", "Current_World_Pay_Token"]
yn_cols = [c for c in yn_cols if c in df.columns]
for col in yn_cols:
    df = df.withColumn(col,
        F.when(F.lower(F.trim(F.col(col))) == "y", 1.0)
         .when(F.lower(F.trim(F.col(col))) == "n", 0.0)
         .otherwise(None).cast(DoubleType())
    )

# ── 5. Parse date columns → days since 2000-01-01 ─────────────────────────
# Handles both native DateType (already in Delta table) and string-encoded dates
from pyspark.sql.types import DateType, TimestampType

date_col_names = ["Renewal_Month", "Registration_Date", "Proforma_Date",
                  "Prospect_Renewal_Date", "Closed_Date", "Last_Renewal",
                  "DateTime_Out"]
date_col_names = [c for c in date_col_names if c in df.columns]

# Also catch any other DateType/TimestampType columns not in the list above
schema_date_cols = [
    f.name for f in df.schema.fields
    if isinstance(f.dataType, (DateType, TimestampType))
    and f.name not in date_col_names
]
all_date_cols = date_col_names + schema_date_cols

epoch = F.lit("2000-01-01").cast("date")
date_types = {f.name: f.dataType for f in df.schema.fields}

for col in all_date_cols:
    dtype = date_types.get(col)
    if isinstance(dtype, (DateType, TimestampType)):
        # Already a date/timestamp — use unix_date (days since unix epoch 1970-01-01)
        # then offset to our epoch (2000-01-01 = day 10957 in unix days)
        df = df.withColumn(
            col + "_days",
            (F.unix_date(F.col(col).cast("date")) - 10957).cast(DoubleType())
        )
    else:
        # String — parse first
        df = df.withColumn(
            col + "_days",
            F.datediff(F.to_date(F.col(col), "dd-MM-yyyy"), epoch).cast(DoubleType())
        )
df = df.drop(*all_date_cols)

# ── 6. Encode low-cardinality categoricals, drop high-cardinality ones ─────
# Uses a plain F.when chain — no ML models, no cache pressure
obj_cols = [f.name for f in df.schema.fields if str(f.dataType) == "StringType()"]
print(f"\nRemaining string columns: {obj_cols}")

for col in obj_cols:
    labels = (
        df.select(col)
          .where(F.col(col).isNotNull())
          .distinct()
          .toPandas()[col]
          .tolist()
    )
    n_unique = len(labels)

    if n_unique <= 20:
        # Build a when-chain: each label → its integer index (0-based)
        mapping = F.when(F.lit(False), F.lit(None).cast(DoubleType()))  # base case
        for idx, label in enumerate(sorted(labels)):
            mapping = mapping.when(F.col(col) == label, F.lit(float(idx)))
        mapping = mapping.otherwise(None)   # nulls and unseen → null

        df = df.withColumn(col + "_idx", mapping).drop(col)
        print(f"  Encoded: {col} ({n_unique} uniques)")
    else:
        print(f"  Dropping high-cardinality column: {col} ({n_unique} uniques)")
        df = df.drop(col)

# ── 7. Cast all remaining columns to Double ────────────────────────────────
# Handle any residual DateType/TimestampType first (unix_date), then cast rest
from pyspark.sql.types import DateType, TimestampType

for field in df.schema.fields:
    if isinstance(field.dataType, (DateType, TimestampType)):
        df = df.withColumn(
            field.name,
            (F.unix_date(F.col(field.name).cast("date")) - 10957).cast(DoubleType())
        )
    elif str(field.dataType) != "DoubleType()":
        df = df.withColumn(field.name, F.col(field.name).cast(DoubleType()))

# ── 8. Drop rows where ALL values are null ────────────────────────────────
df = df.dropna(how="all")
print(f"\nShape after cleaning: ({df.count()}, {len(df.columns)})")

# ── 9. Correlation matrix via pandas (collect to driver) ──────────────────
# Safe for ~60 columns; if you have thousands of cols, sample first
pdf: pd.DataFrame = df.toPandas()
pdf.dropna(axis=1, how="all", inplace=True)          # drop any all-null cols
corr = pdf.corr(method="pearson")
print(f"\nCorrelation matrix size: {corr.shape}")

# ── 10. Highlight target column ───────────────────────────────────────────
TARGET = "churned"                # binary: 0=retained, 1=churned

# Reorder so target is first (makes it easy to read across the top/left)
if TARGET in corr.columns:
    cols_ordered = [TARGET] + [c for c in corr.columns if c != TARGET]
    corr = corr.loc[cols_ordered, cols_ordered]

# ── 11. Plot — full matrix, annotated ─────────────────────────────────────
n = corr.shape[0]
fig_size = max(16, n * 0.55)

fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

sns.heatmap(
    corr,
    mask=None,                      # no mask — full matrix
    annot=True,                     # always show values
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.4,
    square=True,
    ax=ax,
    annot_kws={"size": 7},
)

# Highlight the target row/col with a bold border
if TARGET in corr.columns:
    idx = corr.columns.tolist().index(TARGET)
    ax.add_patch(plt.Rectangle((0, idx), n, 1, fill=False, edgecolor="black", lw=2.5, clip_on=False))
    ax.add_patch(plt.Rectangle((idx, 0), 1, n, fill=False, edgecolor="black", lw=2.5, clip_on=False))

ax.set_title("Billings — Pearson Correlation Matrix  (target: churned, 28-day window)", fontsize=14, pad=12)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()

plt.savefig("/tmp/corr_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 12. Target correlations ranked ────────────────────────────────────────
if TARGET in corr.columns:
    target_corr = (
        corr[[TARGET]]
        .drop(index=TARGET)
        .rename(columns={TARGET: "correlation_with_churned"})
        .assign(abs_corr=lambda d: d["correlation_with_churned"].abs())
        .sort_values("abs_corr", ascending=False)
        .drop(columns="abs_corr")
    )
    print("\nFeature correlations with churned (ranked):")
    print(target_corr.to_string())
    display(spark.createDataFrame(target_corr.reset_index().rename(columns={"index": "feature"})))

# ── 13. Save labelled set for model training ──────────────────────────────
# Uncomment to persist
# df_labelled_clean = spark.createDataFrame(pdf)  # if you want the cleaned version
# df_labelled_clean.write.mode("overwrite").saveAsTable("`ds-raw-datasets`.`billings_churn_labelled`")

# COMMAND ----------

FEATURES = [
    # Target
    "churned",

    # ID / traceability
    "co_ref",

    # Dates (raw) + datediff
    "prospect_renewal_date",
    "closed_date",
    "datediff",

    # Top features from Mann-Whitney
    "total_renewal_score_new",
    "status_scores",
    "sustainability_score",
    "auto_renewal_score",
    "renewal_score_at_release",
    "anchoring_score",
    "current_anchorings",       # keeping over proforma_approved_lists
    "tenure_years",
    "registration_date_days",   # keep as numeric (days since 2000-01-01)
    "last_years_price",         # representative of the price cluster
]

df_final = (
    df_raw
    .filter(F.col("prospect_outcome") != "Open")
    .withColumn("datediff", F.datediff("closed_date", "prospect_renewal_date"))
    .filter((F.col("datediff") >= 0) & (F.col("datediff") < 29))
    .withColumn(
        "churned",
        F.when(F.col("prospect_outcome") == "Won", 0.0).otherwise(1.0)
    )
    # registration_date → numeric (days since 2000-01-01)
    .withColumn(
        "registration_date_days",
        F.datediff(
            F.to_date(F.col("registration_date"), "dd-MM-yyyy"),
            F.lit("2000-01-01").cast("date")
        ).cast(DoubleType())
    )
    .select(FEATURES)
    .dropna(subset=[c for c in FEATURES if c not in ("co_ref", "datediff", "prospect_renewal_date", "closed_date")])
)

print(f"Final dataframe shape: ({df_final.count()}, {len(df_final.columns)})")
print(f"Churn rate: {df_final.filter(F.col('churned') == 1).count() / df_final.count():.1%}")
# save the df
df_final.write.mode("overwrite").saveAsTable("`ds-raw-datasets`.`billings_churn_final`")

# COMMAND ----------

df.columns