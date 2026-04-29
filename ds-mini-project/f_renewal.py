# Databricks notebook source
from pyspark.sql.functions import DataFrame
from pyspark.sql import functions as F
import pandas as pd

# COMMAND ----------

df_renewal: DataFrame = spark.table("workspace.`ds-raw-datasets`.raw_renewal_calls")
df_renewal.display()

# COMMAND ----------

pd_renewal: pd.DataFrame = df_renewal.toPandas()
pd_renewal.columns

# COMMAND ----------

ren_pd = pd_renewal.dropna(subset=['co_ref', "call_date"])
ren_pd.groupby(['co_ref']).agg({'call_date': 'max'}, columns=['call_date']).sort_values(by='call_date', ascending=False).display()
# print(ren_pd.shape, pd_renewal.shape)

# COMMAND ----------

# MAGIC %sql
# MAGIC use workspace.`ds-raw-datasets`;
# MAGIC
# MAGIC with renewal_calls as (
# MAGIC   select co_ref, max(call_date) as call_date from raw_renewal_calls group by co_ref
# MAGIC ) select * from f_billings b join renewal_calls r on b.co_ref = r.co_ref and r.call_date between b.prospect_renewal_date and b.closed_date order by b.co_ref, r.call_date

# COMMAND ----------

df_renewal.select('customer_renewal_response_category').distinct().show()

# COMMAND ----------

# MAGIC %sql
# MAGIC use workspace.`ds-raw-datasets`;
# MAGIC select co_ref, call_date, customer_renewal_response_category from raw_renewal_calls r where co_ref is not null group by co_ref, call_date, customer_renewal_response_category having customer_renewal_response_category is not null and customer_renewal_response_category not like '%Not Mentioned%' order by co_ref, call_date;

# COMMAND ----------

df_cc: DataFrame = spark.table("workspace.`ds-raw-datasets`.raw_cc_calls")
df_cc.display()