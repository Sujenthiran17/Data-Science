# Databricks notebook source
# DBTITLE 1,Imports
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import pandas as pd
import numpy as np
from pyspark.sql.dataframe import DataFrame
from datetime import datetime

# COMMAND ----------

# DBTITLE 1,Spark Init
spark: SparkSession = SparkSession.builder.getOrCreate()

# COMMAND ----------

# DBTITLE 1,Loading DataFrames
# lets see what the billing has to do?
df_billings: DataFrame = spark.table("workspace.`ds-raw-datasets`.raw_billings")
df_renewals: DataFrame = spark.table("workspace.`ds-raw-datasets`.raw_renewal_calls")
df_cc_calls: DataFrame = spark.table("workspace.`ds-raw-datasets`.raw_cc_calls")
df_emails:   DataFrame = spark.table("workspace.`ds-raw-datasets`.raw_emails")

# COMMAND ----------

# DBTITLE 1,Dropping Columns
"""
Dropping 'proforma_auto_renewal' cause it has no impact on the outcome
proforma_auto_renewal	prospect_outcome	count
        true	            Won	            45947
        false	            Won	            483
        true	            Churned	        7323
        false	            Churned	        209
        null	            Won	            5887
        null	            Churned	        2120
"""
drop_columns = [
    "proforma_auto_renewal", "last_years_date_paid", "proforma_date",
    "registration_date", "payment_timeframe", "proforma_audit_status",
    "renewal_score_at_release", # its like feeding previous prediction score to the model,
    "proforma_approved_lists", 'current_anchor_list', 'prospect_status',
    "starting_net", "total_net_paid", "last_renewal", "last_band",
    "last_total_net_paid", "last_connections", "renewal_year",
    "date_time_out", "last_years_price", "proforma_world_pay_token"
]

# The last_years_date_paid column is fully null - drop it
df_billings = df_billings.drop(*drop_columns)

# COMMAND ----------

# DBTITLE 1,Handling Null Data
# Understand the billings database
df_billings = df_billings.dropna(subset=["co_ref"])

# Drop the records that dont have closed date
# "connection_group", "of_connection", "anchor_group" all have 65 null values
# in the same record. So, remove those rows.
df_billings = df_billings.dropna(subset=["closed_date", "of_connection"])

zero_fillers = [
    'connection_net', 'connection_qty', 'starting_connection_net',
    'starting_connection_qty', 'tenure_years'
]
df_billings = df_billings.fillna(0, subset=zero_fillers)
df_billings = df_billings.fillna("0", subset=["tenure_group"])

df_billings = df_billings.fillna("0.0%", subset=['discount_amount'])
df_billings = df_billings.withColumn(
    "discount_amount",
    (F.regexp_replace(F.col("discount_amount"), "%", ""))
)
df_billings = df_billings.withColumn(
    "tenure_group",
    (F.regexp_replace(F.col("tenure_group"), r"\+", ""))
)

unknown_fillers = [
    "proforma_account_stage",
    "proforma_membership_status",
    "band"
]
df_billings = df_billings.fillna('unknown', subset=unknown_fillers)

# COMMAND ----------

# DBTITLE 1,Handling the dates
# Changing open to churned
df_billings = df_billings.withColumn(
    "prospect_outcome",
    F.when(
        (F.col("prospect_outcome") == "Open") &
        (F.datediff(F.current_date(), F.col("prospect_renewal_date")) > 27),
        "Churned"
    ).otherwise(F.col("prospect_outcome"))
)

# Create date diff and filter out the pre renual records.
df_billings = df_billings.withColumn("days_to_close", F.datediff(F.col("closed_date"), F.col("prospect_renewal_date")))
df_billings = df_billings.filter((F.col("prospect_outcome") != "Open") & (F.col("days_to_close") >= 0))

# COMMAND ----------

# DBTITLE 1,Type Casting
df_billings = df_billings.withColumn(
    "sustainability_score",
    F.col("sustainability_score").cast("float")
)
df_billings = df_billings.withColumn(
    "total_renewal_score_new",
    F.col("total_renewal_score_new").cast("float")
)
df_billings = df_billings.withColumn(
    "discount_amount",
    F.col("discount_amount").cast("float")
)
df_billings = df_billings.withColumn(
    "auto_renewal_score",
    F.col("auto_renewal_score").cast("int")
)
df_billings = df_billings.withColumn(
    "status_scores",
    F.col("status_scores").cast("int")
)
df_billings = df_billings.withColumn(
    "anchoring_score",
    F.col("anchoring_score").cast("float")
)
df_billings = df_billings.withColumn(
    "tenure_scores",
    F.col("tenure_scores").cast("float")
)
df_billings = df_billings.withColumn(
    "tenure_group",
    F.col("tenure_group").cast("int")
)
df_billings = df_billings.withColumn(
    "proforma_account_stage",
    F.lower(F.trim(F.col("proforma_account_stage")))
)

# COMMAND ----------

# DBTITLE 1,Store & Display Table
df_billings.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.`ds-datasets`.billings")
display(df_billings)