# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ── 1. Load / prepare billings ──────────────────────────────────────────
df_billings = spark.table("workspace.`ds-raw-datasets`.raw_billings") \
    .filter(F.col("prospect_outcome") != "Open") \
    .withColumn("datediff", F.datediff("closed_date", "prospect_renewal_date")) \
    .filter(F.col("datediff") < 29) \
    .withColumn("index", F.monotonically_increasing_id())

df_billings = df_billings.select(
    "index",
    "co_ref",
    "prospect_renewal_date",
    "closed_date",
    "datediff",
    "total_renewal_score_new",
    "status_scores",
    "sustainability_score",
    "auto_renewal_score",
    "renewal_score_at_release",
    "anchoring_score",
    "current_anchorings",
    "tenure_years",
    "last_years_price",
    "prospect_outcome"
).withColumn("prospect_renewal_date", F.to_date("prospect_renewal_date")) \
 .withColumn("closed_date", F.to_date("closed_date"))

call_cols = [
    "co_ref",
    "call_date",
    "call_direction",
    "customer_reaction_category",
    "agent_renewal_pitch_category",
    "customer_renewal_response_category",
    "membership_renewal_decision",
    "serious_complaint",
    "other_complaint",
    "discussion_on_price_increase",
    "renewal_impact_due_to_price_increase",
    "discount_or_waiver_requested",
    "discount_offered",
    "explicit_competitor_mention",
    "explicit_switching_intent",
    "desire_to_cancel",
    "customer_response",
    "agent_renewal_initiation",
    "call_number"
]

df_calls = spark.table("workspace.`ds-raw-datasets`.raw_renewal_calls") \
    .select(*call_cols + ["analysed_call"]) \
    .filter(F.col("analysed_call") == "1") \
    .drop("analysed_call") \
    .withColumn("call_date", F.to_date("call_date")) \
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

# COMMAND ----------

df_latest_call.toPandas().notnull().sum()