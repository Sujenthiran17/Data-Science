# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ── 1. Load / prepare billings ──────────────────────────────────────────
df_billings = spark.table("workspace.`ds-raw-datasets`.raw_billings") \
    .filter(F.col("prospect_outcome") != "Open") \
    .withColumn("datediff", F.datediff("closed_date", "prospect_renewal_date")) \
    .filter(F.col("datediff") < 29) \
    .withColumn("index", F.monotonically_increasing_id())


# Select all columns from raw_renewal_calls
df_calls = spark.table("workspace.`ds-raw-datasets`.raw_renewal_calls") \
    .filter(F.col("analysed_call") == "1") \
    .filter(F.col("customer_renewal_response_category") != "null") \
    .filter(F.col("customer_renewal_response_category") != "Not Mentioned")

# ── 3. Join on co_ref — explicit condition to avoid ambiguous reference ──
df_joined = df_billings.join(
    df_calls,
    on=df_billings["co_ref"] == df_calls["co_ref"],
    how="left"
).filter(
    (F.col("call_date") >= F.col("prospect_renewal_date")) &
    (F.col("call_date") <= F.col("closed_date"))
).drop(df_calls["co_ref"])

# ── 4. Keep only the LATEST call per billing row (index) ────────────────
window = Window.partitionBy("index").orderBy(F.desc("call_date"))

df_latest_call = df_joined \
    .withColumn("rn", F.row_number().over(window)) \
    .filter(F.col("rn") == 1) \
    .drop("rn")

df_latest_call.display()
df_final = df_latest_call
df_final.write.mode("overwrite").saveAsTable("workspace.`ds-raw-datasets`.joined_two_tables")

# COMMAND ----------

print(f"Number of columns: {len(df_latest_call.columns)}")
print(f"Number of rows: {df_latest_call.count()}")

# COMMAND ----------

df_latest_call.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #Cleaning

# COMMAND ----------

# MAGIC %md
# MAGIC ##1.Coref column

# COMMAND ----------

df=df_latest_call

# COMMAND ----------

df.filter(F.col("co_ref").isNull()).count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.renewal Month column

# COMMAND ----------

df.filter(F.col('renewal_month').isNull()).count()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

df = df.withColumn('renewal_month', F.to_date(F.col('renewal_month'), 'dd-MM-yyyy'))
display(df)

# COMMAND ----------

display(df.select('renewal_month').distinct())

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.

# COMMAND ----------

# MAGIC %md
# MAGIC ##3.connection_net

# COMMAND ----------

df.select('connection_net').distinct().show()

# COMMAND ----------

df.filter(F.col('renewal_month').isNull()).count()

# COMMAND ----------

display(df.columns)

# COMMAND ----------

display(null_counts := df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]))

# COMMAND ----------

# MAGIC %md
# MAGIC # Clean All Null Try 1

# COMMAND ----------

# ============================================================
# Null Imputation — Churn Prediction Dataset
# Only columns confirmed to have nulls are handled below
# ============================================================

from pyspark.sql import functions as F
from pyspark.sql import Window


# ────────────────────────────────────────────────────────────
# HELPER — fill nulls with median grouped by segment
# ────────────────────────────────────────────────────────────
def fill_median_by_group(df, target_col, group_cols):
    group_median = (
        df.groupBy(*group_cols)
        .agg(F.percentile_approx(target_col, 0.5).alias("_med"))
    )
    df = df.join(group_median, on=group_cols, how="left")

    global_median = df.select(
        F.percentile_approx(target_col, 0.5)
    ).collect()[0][0]

    df = df.withColumn(
        target_col,
        F.when(
            F.col(target_col).isNull(),
            F.coalesce(F.col("_med"), F.lit(global_median))
        ).otherwise(F.col(target_col))
    ).drop("_med")

    return df


# ────────────────────────────────────────────────────────────
# 1. NUMERIC — connection & quantity fields (7,334 nulls each)
# ────────────────────────────────────────────────────────────
for col in ["connection_net", "connection_qty",
            "starting_connection_net", "starting_connection_qty"]:
    df = fill_median_by_group(df, col, ["connection_group"])


# ────────────────────────────────────────────────────────────
# 2. NUMERIC — discount_amount (6,509 nulls)
# Missing = no discount applied
# ────────────────────────────────────────────────────────────
df = df.withColumn("discount_amount", F.coalesce(F.col("discount_amount"), F.lit(0)))


# ────────────────────────────────────────────────────────────
# 3. DATE — last_years_date_paid (7,905 nulls)
# Proxy: last_renewal − 365 days
# ────────────────────────────────────────────────────────────
df = df.withColumn(
    "last_years_date_paid",
    F.when(
        F.col("last_years_date_paid").isNull() & F.col("last_renewal").isNotNull(),
        F.date_sub(F.col("last_renewal"), 365)
    ).otherwise(F.col("last_years_date_paid"))
)


# ────────────────────────────────────────────────────────────
# 4. NUMERIC — last_years_price (719 nulls)
# ────────────────────────────────────────────────────────────
df = fill_median_by_group(df, "last_years_price", ["renewal_year", "band"])


# ────────────────────────────────────────────────────────────
# 5. PROFORMA fields (7 nulls each) — categorical → "Unknown"
# ────────────────────────────────────────────────────────────
for col in ["proforma_auto_renewal", "proforma_world_pay_token",
            "proforma_account_stage", "proforma_audit_status",
            "proforma_membership_status", "proforma_approved_lists"]:
    df = df.withColumn(col, F.coalesce(F.col(col), F.lit("Unknown")))

# proforma_date (18 nulls) — flag only, do not impute a date
df = df.withColumn(
    "proforma_date_is_null",
    F.when(F.col("proforma_date").isNull(), 1).otherwise(0)
)


# ────────────────────────────────────────────────────────────
# 6. current_anchor_list (1,943 nulls) — list field → "None"
# ────────────────────────────────────────────────────────────
df = df.withColumn(
    "current_anchor_list",
    F.coalesce(F.col("current_anchor_list"), F.lit("None"))
)


# ────────────────────────────────────────────────────────────
# 7. payment_timeframe (1,127 nulls) — mode by band
# ────────────────────────────────────────────────────────────
mode_payment = (
    df.dropna(subset=["payment_timeframe"])
    .groupBy("band", "payment_timeframe")
    .count()
    .withColumn("rn", F.row_number().over(
        Window.partitionBy("band").orderBy(F.col("count").desc())
    ))
    .filter(F.col("rn") == 1)
    .select("band", F.col("payment_timeframe").alias("_mode_payment"))
)
df = df.join(mode_payment, on="band", how="left")
df = df.withColumn(
    "payment_timeframe",
    F.coalesce(F.col("payment_timeframe"), F.col("_mode_payment"))
).drop("_mode_payment")


# ────────────────────────────────────────────────────────────
# 8. registration_date (69 nulls), tenure_years (69 nulls)
# Derive tenure_years from registration_date where possible
# ────────────────────────────────────────────────────────────
df = df.withColumn(
    "tenure_years",
    F.when(
        F.col("tenure_years").isNull() & F.col("registration_date").isNotNull(),
        F.round(F.datediff(F.current_date(), F.col("registration_date")) / 365.25, 1)
    ).otherwise(F.col("tenure_years"))
)
# Remaining tenure_years nulls → median by band
df = fill_median_by_group(df, "tenure_years", ["band"])

# registration_date — flag only, do not fabricate
df = df.withColumn(
    "registration_date_is_null",
    F.when(F.col("registration_date").isNull(), 1).otherwise(0)
)


# ────────────────────────────────────────────────────────────
# 9. total_net_paid (1,127 nulls)
# ────────────────────────────────────────────────────────────
df = fill_median_by_group(df, "total_net_paid", ["renewal_year", "band"])


# ────────────────────────────────────────────────────────────
# 10. of_connection (7 nulls), connection_group (7 nulls)
# ────────────────────────────────────────────────────────────
df = fill_median_by_group(df, "of_connection", ["connection_group"])
df = df.withColumn(
    "connection_group",
    F.coalesce(F.col("connection_group"), F.lit("Unknown"))
)


# ────────────────────────────────────────────────────────────
# 11. last_renewal (1,526), last_band (1,531),
#     last_total_net_paid (1,532), last_connections (1,540)
# ────────────────────────────────────────────────────────────
# last_renewal — reconstruct from renewal_year
df = df.withColumn(
    "last_renewal",
    F.when(
        F.col("last_renewal").isNull() & F.col("renewal_year").isNotNull(),
        F.to_date(F.col("renewal_year"))
    ).otherwise(
        F.coalesce(
            F.to_date(F.col("last_renewal"), "dd-MM-yyyy"),
            F.to_date(F.col("last_renewal"), "yyyy-MM-dd")
        )
    )
)

# last_band — fall back to current band
df = df.withColumn("last_band", F.coalesce(F.col("last_band"), F.col("band")))

df = fill_median_by_group(df, "last_total_net_paid", ["renewal_year", "band"])
df = fill_median_by_group(df, "last_connections",    ["connection_group"])


# ────────────────────────────────────────────────────────────
# 12. anchor_group (7 nulls)
# ────────────────────────────────────────────────────────────
df = df.withColumn(
    "anchor_group",
    F.coalesce(F.col("anchor_group"), F.lit("Unknown"))
)


# ────────────────────────────────────────────────────────────
# 13. CALL / NLP — binary event flags → fill 0
# (serious_complaint=466, other_complaint=466,
#  discussion_on_price_increase=39, etc.)
# ────────────────────────────────────────────────────────────
binary_call_cols = [
    "serious_complaint", "other_complaint",
    "discussion_on_price_increase", "renewal_impact_due_to_price_increase",
    "discount_or_waiver_requested", "call_reschedule_request",
    "agent_flagged_membership_status_alert", "agent_renewal_initiation",
    "explicit_competitor_mention", "explicit_switching_intent",
    "price_switching_mentioned", "competitor_value_comparison",
    "percentage_price_increase_mentioned", "monetary_price_increase_mentioned",
    "price_range_mentioned", "customer_asked_for_justification",
    "desire_to_cancel", "discount_offered", "analysed_call", "c20"
]
for col in binary_call_cols:
    df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0)))


# ────────────────────────────────────────────────────────────
# 14. CALL / NLP — categorical fields → "Not recorded"
# (churn_category=6902, complaint_category=5945, etc.)
# ────────────────────────────────────────────────────────────
not_recorded_cols = [
    "churn_category", "complaint_category",
    "customer_reaction_category", "agent_renewal_pitch_category",
    "customer_renewal_response_category", "agent_response_category",
    "membership_renewal_decision", "topic_introduced_by",
    "customer_response", "call_direction"
]
for col in not_recorded_cols:
    df = df.withColumn(col, F.coalesce(F.col(col), F.lit("Not recorded")))


# ────────────────────────────────────────────────────────────
# 15. CALL / NLP — conditional categories → "Not applicable"
# (justification_category=6166, reason_for_renewal=5268, etc.)
# ────────────────────────────────────────────────────────────
not_applicable_cols = [
    "justification_category", "reason_for_renewal_category",
    "agent_response_to_cancel_category",
    "argument_that_convinced_customer_to_stay_category"
]
for col in not_applicable_cols:
    df = df.withColumn(col, F.coalesce(F.col(col), F.lit("Not applicable")))


# ────────────────────────────────────────────────────────────
# 16. TEXT fields — mentioned_competitors (196 nulls),
#     competitor_benefits_mentioned (2 nulls)
# ────────────────────────────────────────────────────────────
for col in ["mentioned_competitors", "competitor_benefits_mentioned"]:
    df = df.withColumn(col, F.coalesce(F.col(col), F.lit("None")))


# ────────────────────────────────────────────────────────────
# 17. has_world_pay_token — useful signal for modelling
# ────────────────────────────────────────────────────────────
df = df.withColumn(
    "has_world_pay_token",
    F.when(F.col("current_world_pay_token").isNotNull(), 1).otherwise(0)
)


# ────────────────────────────────────────────────────────────
# VERIFY — confirm nulls are resolved
# ────────────────────────────────────────────────────────────
print("=== Null check after imputation ===")
display(df.select([
    F.count(F.when(F.col(c).isNull(), 1)).alias(c)
    for c in df.columns
]))

# COMMAND ----------

# MAGIC %md
# MAGIC # Clean All Null Try 2

# COMMAND ----------

# ============================================================
# Null Imputation — Churn Prediction Dataset (Optimised)
# ============================================================
# Key improvements over v1:
#   • fill_median_by_group: single join per call (no collect + lit round-trip)
#   • Date parsing: try_to_date() instead of to_date() — returns null on
#     malformed input instead of throwing, so coalesce() works correctly
#   • last_renewal reconstruction: safe multi-format parse in one withColumn
#   • Binary / categorical fills: single select() + alias trick to avoid
#     N separate withColumn passes (each pass re-serialises the whole plan)
#   • mode_payment: unchanged logic, but comment added for clarity
# ============================================================

from pyspark.sql import functions as F
from pyspark.sql import Window


# ────────────────────────────────────────────────────────────
# HELPER — fill nulls with median grouped by segment
# Uses a broadcast-friendly single join; avoids a second
# collect() call by letting coalesce handle the global fallback
# inside the join result.
# ────────────────────────────────────────────────────────────
def fill_median_by_group(df, target_col, group_cols):
    group_median = (
        df.groupBy(*group_cols)
          .agg(F.percentile_approx(target_col, 0.5).alias("_med"))
    )
    # Global median computed once and broadcast via lit()
    global_median = (
        df.select(F.percentile_approx(target_col, 0.5).alias("g"))
          .collect()[0][0]
    )
    df = (
        df.join(F.broadcast(group_median), on=group_cols, how="left")
          .withColumn(
              target_col,
              F.when(
                  F.col(target_col).isNull(),
                  F.coalesce(F.col("_med"), F.lit(global_median))
              ).otherwise(F.col(target_col))
          )
          .drop("_med")
    )
    return df


# ────────────────────────────────────────────────────────────
# Batch helper — fill many columns at once in a single
# logical plan step instead of N separate withColumn calls.
# ────────────────────────────────────────────────────────────
def fill_constant(df, cols, value):
    return df.select(
        *[
            F.coalesce(F.col(c), F.lit(value)).alias(c)
            if c in cols else F.col(c)
            for c in df.columns
        ]
    )


# ────────────────────────────────────────────────────────────
# 1. NUMERIC — connection & quantity fields (7,334 nulls each)
# ────────────────────────────────────────────────────────────
for col in [
    "connection_net", "connection_qty",
    "starting_connection_net", "starting_connection_qty",
]:
    df = fill_median_by_group(df, col, ["connection_group"])


# ────────────────────────────────────────────────────────────
# 2. NUMERIC — discount_amount (6,509 nulls)
# Missing = no discount applied → 0
# ────────────────────────────────────────────────────────────
df = df.withColumn(
    "discount_amount",
    F.coalesce(F.col("discount_amount"), F.lit(0))
)


# ────────────────────────────────────────────────────────────
# 3. DATE — last_years_date_paid (7,905 nulls)
# Proxy: last_renewal − 365 days
# ────────────────────────────────────────────────────────────
df = df.withColumn(
    "last_years_date_paid",
    F.when(
        F.col("last_years_date_paid").isNull() & F.col("last_renewal").isNotNull(),
        F.date_sub(F.col("last_renewal"), 365)
    ).otherwise(F.col("last_years_date_paid"))
)


# ────────────────────────────────────────────────────────────
# 4. NUMERIC — last_years_price (719 nulls)
# ────────────────────────────────────────────────────────────
df = fill_median_by_group(df, "last_years_price", ["renewal_year", "band"])


# ────────────────────────────────────────────────────────────
# 5. PROFORMA fields (7 nulls each) — categorical → "Unknown"
# Single plan pass via fill_constant()
# ────────────────────────────────────────────────────────────
proforma_cat_cols = [
    "proforma_auto_renewal", "proforma_world_pay_token",
    "proforma_account_stage", "proforma_audit_status",
    "proforma_membership_status", "proforma_approved_lists",
]
df = fill_constant(df, proforma_cat_cols, "Unknown")

# proforma_date (18 nulls) — flag only, do not impute a date
df = df.withColumn(
    "proforma_date_is_null",
    F.when(F.col("proforma_date").isNull(), 1).otherwise(0)
)


# ────────────────────────────────────────────────────────────
# 6. current_anchor_list (1,943 nulls) — list field → "None"
# ────────────────────────────────────────────────────────────
df = df.withColumn(
    "current_anchor_list",
    F.coalesce(F.col("current_anchor_list"), F.lit("None"))
)


# ────────────────────────────────────────────────────────────
# 7. payment_timeframe (1,127 nulls) — mode by band
# ────────────────────────────────────────────────────────────
mode_payment = (
    df.dropna(subset=["payment_timeframe"])
      .groupBy("band", "payment_timeframe")
      .count()
      .withColumn(
          "rn",
          F.row_number().over(
              Window.partitionBy("band").orderBy(F.col("count").desc())
          )
      )
      .filter(F.col("rn") == 1)
      .select("band", F.col("payment_timeframe").alias("_mode_payment"))
)
df = (
    df.join(F.broadcast(mode_payment), on="band", how="left")
      .withColumn(
          "payment_timeframe",
          F.coalesce(F.col("payment_timeframe"), F.col("_mode_payment"))
      )
      .drop("_mode_payment")
)


# ────────────────────────────────────────────────────────────
# 8. registration_date (69 nulls), tenure_years (69 nulls)
# ────────────────────────────────────────────────────────────
df = df.withColumn(
    "tenure_years",
    F.when(
        F.col("tenure_years").isNull() & F.col("registration_date").isNotNull(),
        F.round(
            F.datediff(F.current_date(), F.col("registration_date")) / 365.25, 1
        )
    ).otherwise(F.col("tenure_years"))
)
df = fill_median_by_group(df, "tenure_years", ["band"])

df = df.withColumn(
    "registration_date_is_null",
    F.when(F.col("registration_date").isNull(), 1).otherwise(0)
)


# ────────────────────────────────────────────────────────────
# 9. total_net_paid (1,127 nulls)
# ────────────────────────────────────────────────────────────
df = fill_median_by_group(df, "total_net_paid", ["renewal_year", "band"])


# ────────────────────────────────────────────────────────────
# 10. of_connection (7 nulls), connection_group (7 nulls)
# Note: fill of_connection BEFORE connection_group to keep the
# group key intact for the join.
# ────────────────────────────────────────────────────────────
df = fill_median_by_group(df, "of_connection", ["connection_group"])
df = df.withColumn(
    "connection_group",
    F.coalesce(F.col("connection_group"), F.lit("Unknown"))
)


# ────────────────────────────────────────────────────────────
# 11. last_renewal (1,526), last_band (1,531),
#     last_total_net_paid (1,532), last_connections (1,540)
#
# FIX: the original code used coalesce(to_date(...fmt1),
#      to_date(...fmt2)) which throws a SparkException on
#      malformed input BEFORE coalesce can catch the null.
#
#      Use try_to_date() (Spark 3.4+) or regexp-guard the
#      format choice:
#        • if value matches yyyy-MM-dd  → parse as-is
#        • otherwise try dd-MM-yyyy     → safe fallback
# ────────────────────────────────────────────────────────────

def safe_parse_date(col_name):
    """
    Returns a Column expression that parses date strings robustly
    without throwing on malformed input.

    Spark 3.4+: try_to_date() returns null for unparseable values.
    Older Spark: the regexp branch detects the format first.
    """
    c = F.col(col_name)
    try:
        # Preferred: try_to_date silently returns null on bad input
        return F.coalesce(
            F.try_to_date(c, "yyyy-MM-dd"),
            F.try_to_date(c, "dd-MM-yyyy"),
        )
    except AttributeError:
        # Fallback for Spark < 3.4
        iso_pattern = r"^\d{4}-\d{2}-\d{2}$"
        return F.when(
            c.rlike(iso_pattern),
            F.to_date(c, "yyyy-MM-dd")
        ).otherwise(
            F.to_date(c, "dd-MM-yyyy")   # returns null on mismatch with PERMISSIVE
        )


df = df.withColumn(
    "last_renewal",
    F.when(
        F.col("last_renewal").isNull() & F.col("renewal_year").isNotNull(),
        F.to_date(F.col("renewal_year").cast("string"), "yyyy")
    ).otherwise(safe_parse_date("last_renewal"))
)

df = df.withColumn(
    "last_band",
    F.coalesce(F.col("last_band"), F.col("band"))
)

df = fill_median_by_group(df, "last_total_net_paid", ["renewal_year", "band"])
df = fill_median_by_group(df, "last_connections",    ["connection_group"])


# ────────────────────────────────────────────────────────────
# 12. anchor_group (7 nulls)
# ────────────────────────────────────────────────────────────
df = df.withColumn(
    "anchor_group",
    F.coalesce(F.col("anchor_group"), F.lit("Unknown"))
)


# ────────────────────────────────────────────────────────────
# 13. CALL / NLP — binary event flags → fill 0
# Single plan pass via fill_constant()
# ────────────────────────────────────────────────────────────
binary_call_cols = [
    "serious_complaint", "other_complaint",
    "discussion_on_price_increase", "renewal_impact_due_to_price_increase",
    "discount_or_waiver_requested", "call_reschedule_request",
    "agent_flagged_membership_status_alert", "agent_renewal_initiation",
    "explicit_competitor_mention", "explicit_switching_intent",
    "price_switching_mentioned", "competitor_value_comparison",
    "percentage_price_increase_mentioned", "monetary_price_increase_mentioned",
    "price_range_mentioned", "customer_asked_for_justification",
    "desire_to_cancel", "discount_offered", "analysed_call", "c20",
]
df = fill_constant(df, binary_call_cols, 0)


# ────────────────────────────────────────────────────────────
# 14. CALL / NLP — categorical fields → "Not recorded"
# ────────────────────────────────────────────────────────────
not_recorded_cols = [
    "churn_category", "complaint_category",
    "customer_reaction_category", "agent_renewal_pitch_category",
    "customer_renewal_response_category", "agent_response_category",
    "membership_renewal_decision", "topic_introduced_by",
    "customer_response", "call_direction",
]
df = fill_constant(df, not_recorded_cols, "Not recorded")


# ────────────────────────────────────────────────────────────
# 15. CALL / NLP — conditional categories → "Not applicable"
# ────────────────────────────────────────────────────────────
not_applicable_cols = [
    "justification_category", "reason_for_renewal_category",
    "agent_response_to_cancel_category",
    "argument_that_convinced_customer_to_stay_category",
]
df = fill_constant(df, not_applicable_cols, "Not applicable")


# ────────────────────────────────────────────────────────────
# 16. TEXT fields — mentioned_competitors, competitor_benefits
# ────────────────────────────────────────────────────────────
df = fill_constant(
    df,
    ["mentioned_competitors", "competitor_benefits_mentioned"],
    "None"
)


# ────────────────────────────────────────────────────────────
# 17. has_world_pay_token — useful signal for modelling
# ────────────────────────────────────────────────────────────
df = df.withColumn(
    "has_world_pay_token",
    F.when(F.col("current_world_pay_token").isNotNull(), 1).otherwise(0)
)


# ────────────────────────────────────────────────────────────
# VERIFY — confirm nulls are resolved
# ────────────────────────────────────────────────────────────
print("=== Null check after imputation ===")
display(
    df.select([
        F.count(F.when(F.col(c).isNull(), 1)).alias(c)
        for c in df.columns
    ])
)

# ONE ETERNITY LATER...

# COMMAND ----------

# MAGIC %md
# MAGIC # Clean All Null And Try 3

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window

# ── STEP 0: Break lineage by writing to a Delta temp table ──
# On serverless, this is the only reliable way to reset a deep plan.
# checkpoint() and sparkContext are both blocked.
df.write.mode("overwrite").saveAsTable("_tmp_churn_imputation_base")
df = spark.table("_tmp_churn_imputation_base")
# Delta read is already optimised — no manual cache() needed on serverless.
# The serverless runtime manages memory automatically.

# ── STEP 1: All medians in one groupBy per key ──────────────
conn_median = (
    df.groupBy("connection_group")
      .agg(
          F.percentile_approx("connection_net",          0.5).alias("_med_connection_net"),
          F.percentile_approx("connection_qty",          0.5).alias("_med_connection_qty"),
          F.percentile_approx("starting_connection_net", 0.5).alias("_med_starting_connection_net"),
          F.percentile_approx("starting_connection_qty", 0.5).alias("_med_starting_connection_qty"),
          F.percentile_approx("#_of_connection",           0.5).alias("_med_of_connection"),
          F.percentile_approx("last_connections",        0.5).alias("_med_last_connections"),
      )
)

yr_band_median = (
    df.groupBy("renewal_year", "band")
      .agg(
          F.percentile_approx("last_years_price",    0.5).alias("_med_last_years_price"),
          F.percentile_approx("total_net_paid",      0.5).alias("_med_total_net_paid"),
          F.percentile_approx("last_total_net_paid", 0.5).alias("_med_last_total_net_paid"),
      )
)

band_median = (
    df.groupBy("band")
      .agg(
          F.percentile_approx("tenure_years", 0.5).alias("_med_tenure_years"),
      )
)

global_medians = df.select(
    F.percentile_approx("connection_net",          0.5).alias("connection_net"),
    F.percentile_approx("connection_qty",          0.5).alias("connection_qty"),
    F.percentile_approx("starting_connection_net", 0.5).alias("starting_connection_net"),
    F.percentile_approx("starting_connection_qty", 0.5).alias("starting_connection_qty"),
    F.percentile_approx("#_of_connection",           0.5).alias("#_of_connection"),
    F.percentile_approx("last_connections",        0.5).alias("last_connections"),
    F.percentile_approx("last_years_price",        0.5).alias("last_years_price"),
    F.percentile_approx("total_net_paid",          0.5).alias("total_net_paid"),
    F.percentile_approx("last_total_net_paid",     0.5).alias("last_total_net_paid"),
    F.percentile_approx("tenure_years",            0.5).alias("tenure_years"),
).collect()[0]

# ── STEP 2: Mode for payment_timeframe ──────────────────────
mode_payment = (
    df.dropna(subset=["payment_timeframe"])
      .groupBy("band", "payment_timeframe")
      .count()
      .withColumn(
          "rn",
          F.row_number().over(
              Window.partitionBy("band").orderBy(F.col("count").desc())
          )
      )
      .filter(F.col("rn") == 1)
      .select("band", F.col("payment_timeframe").alias("_mode_payment"))
)

# ── STEP 3: All joins chained ────────────────────────────────
df = (
    df
    .join(F.broadcast(conn_median),    on="connection_group",       how="left")
    .join(F.broadcast(yr_band_median), on=["renewal_year", "band"], how="left")
    .join(F.broadcast(band_median),    on="band",                   how="left")
    .join(F.broadcast(mode_payment),   on="band",                   how="left")
)

# ── STEP 4: All fills in one select() ───────────────────────
def _med(col_name, lookup_col, global_val):
    return F.when(
        F.col(col_name).isNull(),
        F.coalesce(F.col(lookup_col), F.lit(global_val))
    ).otherwise(F.col(col_name)).alias(col_name)

def _coalesce_lit(col_name, value):
    return F.coalesce(F.col(col_name), F.lit(value)).alias(col_name)

def safe_parse_date(col_name):
    c = F.col(col_name)
    try:
        return F.coalesce(
            F.try_to_date(c, "yyyy-MM-dd"),
            F.try_to_date(c, "dd-MM-yyyy"),
        ).alias(col_name)
    except AttributeError:
        iso_pattern = r"^\d{4}-\d{2}-\d{2}$"
        return F.when(c.rlike(iso_pattern), F.to_date(c, "yyyy-MM-dd")) \
                .otherwise(F.to_date(c, "dd-MM-yyyy")).alias(col_name)

lookup_drop_cols = {
    "_med_connection_net", "_med_connection_qty",
    "_med_starting_connection_net", "_med_starting_connection_qty",
    "_med_of_connection", "_med_last_connections",
    "_med_last_years_price", "_med_total_net_paid", "_med_last_total_net_paid",
    "_med_tenure_years", "_mode_payment",
}

transform_cols = {
    "connection_net", "connection_qty",
    "starting_connection_net", "starting_connection_qty",
    "#_of_connection", "last_connections",
    "last_years_price", "total_net_paid", "last_total_net_paid",
    "tenure_years", "discount_amount", "last_years_date_paid",
    "proforma_auto_renewal", "proforma_world_pay_token",
    "proforma_account_stage", "proforma_audit_status",
    "proforma_membership_status", "proforma_approved_lists",
    "current_anchor_list", "payment_timeframe",
    "last_renewal", "last_band", "anchor_group", "connection_group",
    "mentioned_competitors", "competitor_benefits_mentioned",
    "serious_complaint", "other_complaint",
    "discussion_on_price_increase", "renewal_impact_due_to_price_increase",
    "discount_or_waiver_requested", "call_reschedule_request",
    "agent_flagged_membership_status_alert", "agent_renewal_initiation",
    "explicit_competitor_mention", "explicit_switching_intent",
    "price_switching_mentioned", "competitor_value_comparison",
    "percentage_price_increase_mentioned", "monetary_price_increase_mentioned",
    "price_range_mentioned", "customer_asked_for_justification",
    "desire_to_cancel", "discount_offered", "analysed_call", "c20",
    "churn_category", "complaint_category",
    "customer_reaction_category", "agent_renewal_pitch_category",
    "customer_renewal_response_category", "agent_response_category",
    "membership_renewal_decision", "topic_introduced_by",
    "customer_response", "call_direction",
    "justification_category", "reason_for_renewal_category",
    "agent_response_to_cancel_category",
    "argument_that_convinced_customer_to_stay_category",
}

passthrough = [
    F.col(c) for c in df.columns
    if c not in lookup_drop_cols and c not in transform_cols
]

df = df.select(
    *passthrough,
    _med("connection_net",          "_med_connection_net",          global_medians["connection_net"]),
    _med("connection_qty",          "_med_connection_qty",          global_medians["connection_qty"]),
    _med("starting_connection_net", "_med_starting_connection_net", global_medians["starting_connection_net"]),
    _med("starting_connection_qty", "_med_starting_connection_qty", global_medians["starting_connection_qty"]),
    _med("#_of_connection",           "_med_of_connection",           global_medians["#_of_connection"]),
    _med("last_connections",        "_med_last_connections",        global_medians["last_connections"]),
    _med("last_years_price",    "_med_last_years_price",    global_medians["last_years_price"]),
    _med("total_net_paid",      "_med_total_net_paid",      global_medians["total_net_paid"]),
    _med("last_total_net_paid", "_med_last_total_net_paid", global_medians["last_total_net_paid"]),
    _med("tenure_years",        "_med_tenure_years",        global_medians["tenure_years"]),
    _coalesce_lit("discount_amount",               0),
    _coalesce_lit("current_anchor_list",           "None"),
    _coalesce_lit("anchor_group",                  "Unknown"),
    _coalesce_lit("connection_group",              "Unknown"),
    _coalesce_lit("proforma_auto_renewal",         "Unknown"),
    _coalesce_lit("proforma_world_pay_token",      "Unknown"),
    _coalesce_lit("proforma_account_stage",        "Unknown"),
    _coalesce_lit("proforma_audit_status",         "Unknown"),
    _coalesce_lit("proforma_membership_status",    "Unknown"),
    _coalesce_lit("proforma_approved_lists",       "Unknown"),
    _coalesce_lit("mentioned_competitors",         "None"),
    _coalesce_lit("competitor_benefits_mentioned", "None"),
    _coalesce_lit("serious_complaint",                              0),
    _coalesce_lit("other_complaint",                                0),
    _coalesce_lit("discussion_on_price_increase",                   0),
    _coalesce_lit("renewal_impact_due_to_price_increase",           0),
    _coalesce_lit("discount_or_waiver_requested",                   0),
    _coalesce_lit("call_reschedule_request",                        0),
    _coalesce_lit("agent_flagged_membership_status_alert",          0),
    _coalesce_lit("agent_renewal_initiation",                       0),
    _coalesce_lit("explicit_competitor_mention",                    0),
    _coalesce_lit("explicit_switching_intent",                      0),
    _coalesce_lit("price_switching_mentioned",                      0),
    _coalesce_lit("competitor_value_comparison",                    0),
    _coalesce_lit("percentage_price_increase_mentioned",            0),
    _coalesce_lit("monetary_price_increase_mentioned",              0),
    _coalesce_lit("price_range_mentioned",                          0),
    _coalesce_lit("customer_asked_for_justification",               0),
    _coalesce_lit("desire_to_cancel",                               0),
    _coalesce_lit("discount_offered",                               0),
    _coalesce_lit("analysed_call",                                  0),
    _coalesce_lit("c20",                                            0),
    _coalesce_lit("churn_category",                     "Not recorded"),
    _coalesce_lit("complaint_category",                 "Not recorded"),
    _coalesce_lit("customer_reaction_category",         "Not recorded"),
    _coalesce_lit("agent_renewal_pitch_category",       "Not recorded"),
    _coalesce_lit("customer_renewal_response_category", "Not recorded"),
    _coalesce_lit("agent_response_category",            "Not recorded"),
    _coalesce_lit("membership_renewal_decision",        "Not recorded"),
    _coalesce_lit("topic_introduced_by",                "Not recorded"),
    _coalesce_lit("customer_response",                  "Not recorded"),
    _coalesce_lit("call_direction",                     "Not recorded"),
    _coalesce_lit("justification_category",                             "Not applicable"),
    _coalesce_lit("reason_for_renewal_category",                        "Not applicable"),
    _coalesce_lit("agent_response_to_cancel_category",                  "Not applicable"),
    _coalesce_lit("argument_that_convinced_customer_to_stay_category",  "Not applicable"),
    F.when(
        F.col("last_years_date_paid").isNull() & F.col("last_renewal").isNotNull(),
        F.date_sub(F.col("last_renewal"), 365)
    ).otherwise(F.col("last_years_date_paid")).alias("last_years_date_paid"),
    F.when(
        F.col("last_renewal").isNull() & F.col("renewal_year").isNotNull(),
        F.to_date(F.col("renewal_year").cast("string"), "yyyy")
    ).otherwise(safe_parse_date("last_renewal")).alias("last_renewal"),
    F.coalesce(F.col("last_band"), F.col("band")).alias("last_band"),
    F.coalesce(F.col("payment_timeframe"), F.col("_mode_payment")).alias("payment_timeframe"),
    F.when(F.col("proforma_date").isNull(),              1).otherwise(0).alias("proforma_date_is_null"),
    F.when(F.col("registration_date").isNull(),          1).otherwise(0).alias("registration_date_is_null"),
    F.when(F.col("current_world_pay_token").isNotNull(), 1).otherwise(0).alias("has_world_pay_token"),
)

# ── STEP 5: tenure_years secondary fill ─────────────────────
df = df.withColumn(
    "tenure_years",
    F.when(
        F.col("tenure_years").isNull() & F.col("registration_date").isNotNull(),
        F.round(F.datediff(F.current_date(), F.col("registration_date")) / 365.25, 1)
    ).otherwise(F.col("tenure_years"))
)

# ── STEP 6: Write result & clean up temp table ───────────────
df.write.mode("overwrite").saveAsTable("_tmp_churn_imputed")
df = spark.table("_tmp_churn_imputed")
spark.sql("DROP TABLE IF EXISTS _tmp_churn_imputation_base")

# ── VERIFY ───────────────────────────────────────────────────
print("=== Null check after imputation ===")
display(
    df.select([
        F.count(F.when(F.col(c).isNull(), 1)).alias(c)
        for c in df.columns
    ])
)

# COMMAND ----------

[JVM_ATTRIBUTE_NOT_SUPPORTED] Directly accessing the underlying Spark driver JVM using the attribute 'sparkContext' is not supported on serverless compute. If you require direct access to these fields, consider using a single-user cluster. For more details on compatibility and limitations, check: https://docs.databricks.com/release-notes/serverless.html#limitations