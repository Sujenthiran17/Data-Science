# Databricks notebook source
from pyspark.sql.functions import DataFrame
from pyspark.sql import functions as F

# COMMAND ----------

df_billings: DataFrame = spark.table("`ds-raw-datasets`.`raw_billings`")
df_billings.describe().display()

# COMMAND ----------

df_billings = df_billings\
.filter(F.col("prospect_outcome") != "Open")\
    .withColumn("datediff", F.datediff("closed_date", "prospect_renewal_date"))\
        .filter(F.col("datediff") < 29)

# add an index column to represent each row uniquely
df_billings = df_billings.withColumn("index", F.monotonically_increasing_id())

# COMMAND ----------

df_billings.display()

# COMMAND ----------

f_billings = df_billings.select(
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
)
f_billings.display()

# COMMAND ----------

f_billings.write.mode("overwrite").saveAsTable("`ds-raw-datasets`.`f_billings`")

# COMMAND ----------

