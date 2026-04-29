# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import BooleanType

# ── STEP 0: Break lineage ────────────────────────────────────
df = spark.table("workspace.`ds-raw-datasets`.joined_two_tables")

# ── STEP 0b: Clean percentage-formatted numeric columns ─────
percentage_cols = ["of_connection", "discount_amount"]
for col_name in percentage_cols:
    if col_name in df.columns:
        df = df.withColumn(
            col_name,
            F.regexp_replace(F.col(col_name), r"%", "").cast("double")
        )

# ── STEP 0c: Normalise Yes/No flag columns → INT ─────────────
flag_cols = [
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
for col_name in flag_cols:
    if col_name in df.columns:
        df = df.withColumn(
            col_name,
            F.when(F.upper(F.col(col_name)).isin("YES", "TRUE", "1"), F.lit(1))
             .when(F.upper(F.col(col_name)).isin("NO", "FALSE", "0"), F.lit(0))
             .otherwise(F.lit(None).cast("int"))
        )

# ── STEP 0d: Cast BOOLEAN columns → STRING ───────────────────
bool_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, BooleanType)]
for col_name in bool_cols:
    df = df.withColumn(col_name, F.col(col_name).cast("string"))

# ── STEP 0e: Cast string-fill columns → STRING ───────────────
# Ensures coalesce(col, lit("Unknown")) never tries to cast "Unknown" to BIGINT
string_fill_cols = [
    "proforma_auto_renewal", "proforma_world_pay_token",
    "proforma_account_stage", "proforma_audit_status",
    "proforma_membership_status", "proforma_approved_lists",
    "current_anchor_list", "anchor_group", "connection_group",
    "mentioned_competitors", "competitor_benefits_mentioned",
    "last_band", "payment_timeframe", "call_direction",
    "churn_category", "complaint_category",
    "customer_reaction_category", "agent_renewal_pitch_category",
    "customer_renewal_response_category", "agent_response_category",
    "membership_renewal_decision", "topic_introduced_by",
    "customer_response", "justification_category",
    "reason_for_renewal_category", "agent_response_to_cancel_category",
    "argument_that_convinced_customer_to_stay_category",
]
for col_name in string_fill_cols:
    if col_name in df.columns:
        df = df.withColumn(col_name, F.col(col_name).cast("string"))

# ── STEP 1: All medians in one groupBy per key ──────────────
conn_median = (
    df.groupBy("connection_group")
      .agg(
          F.percentile_approx("connection_net",          0.5).alias("_med_connection_net"),
          F.percentile_approx("connection_qty",          0.5).alias("_med_connection_qty"),
          F.percentile_approx("starting_connection_net", 0.5).alias("_med_starting_connection_net"),
          F.percentile_approx("starting_connection_qty", 0.5).alias("_med_starting_connection_qty"),
          F.percentile_approx("of_connection",           0.5).alias("_med_of_connection"),
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
    F.percentile_approx("of_connection",           0.5).alias("of_connection"),
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

# ── STEP 3b: Pre-parse last_renewal to DATE ──────────────────
df = df.withColumn(
    "_last_renewal_parsed",
    F.coalesce(
        F.try_to_date(F.col("last_renewal"), "yyyy-MM-dd"),
        F.try_to_date(F.col("last_renewal"), "dd-MM-yyyy"),
        F.try_to_date(F.col("last_renewal"), "MM-dd-yyyy"),
    )
)

# ── STEP 4: All fills in one select() ───────────────────────
def _med(col_name, lookup_col, global_val):
    return F.when(
        F.col(col_name).isNull(),
        F.coalesce(F.col(lookup_col), F.lit(global_val))
    ).otherwise(F.col(col_name)).alias(col_name)

def _coalesce_lit(col_name, value):
    return F.coalesce(F.col(col_name), F.lit(value)).alias(col_name)

lookup_drop_cols = {
    "_med_connection_net", "_med_connection_qty",
    "_med_starting_connection_net", "_med_starting_connection_qty",
    "_med_of_connection", "_med_last_connections",
    "_med_last_years_price", "_med_total_net_paid", "_med_last_total_net_paid",
    "_med_tenure_years", "_mode_payment",
    "_last_renewal_parsed",
}

transform_cols = {
    "connection_net", "connection_qty",
    "starting_connection_net", "starting_connection_qty",
    "of_connection", "last_connections",
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
    _med("of_connection",           "_med_of_connection",           global_medians["of_connection"]),
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
        F.col("last_years_date_paid").isNull() & F.col("_last_renewal_parsed").isNotNull(),
        F.date_sub(F.col("_last_renewal_parsed"), 365)
    ).otherwise(F.col("last_years_date_paid")).alias("last_years_date_paid"),
    F.when(
        F.col("last_renewal").isNull() & F.col("renewal_year").isNotNull(),
        F.make_date(F.col("renewal_year"), F.lit(1), F.lit(1))
    ).otherwise(F.col("_last_renewal_parsed")).alias("last_renewal"),
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
    ).otherwise(F.col("tenure_years").cast("INT"))
)

# ── STEP 6: Write result ─────────────────────────────────────
df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.`ds-raw-datasets`.joined_two_tables_removed_nulls")

# ── VERIFY ───────────────────────────────────────────────────
print("=== Null check after imputation ===")
display(df)

# COMMAND ----------

df.count()
df.toPandas().isnull().sum()

# COMMAND ----------


df.filter(F.col("Prospect_Outcome") != "Won").display()

# COMMAND ----------

df_nulls = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
display(df_nulls)