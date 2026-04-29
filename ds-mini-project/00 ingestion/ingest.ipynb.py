# Databricks notebook source
import re
from pyspark.sql.functions import col, sum as _sum, isnan, when

base_path = "s3://ds-miniproject-datasets/"

def to_snake_case(name):
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "", name.strip())
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", cleaned)
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return re.sub(r"_+", "_", snake).lower().strip("_")


def clean_column_names(df):
    return df.toDF(*[to_snake_case(c) for c in df.columns])


def load_excel(spark, path):
    df = (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(path)
    )

    return df


files = {
    "raw_billings": "billings.csv",
    "raw_cc_calls": "cc_calls.csv",
    "raw_emails": "emails.csv",
    "raw_renewal_calls": "renewal_calls.csv",
}

for table_name, filename in files.items():
    df = load_excel(spark, base_path + filename)
    df = clean_column_names(df)

    print(f"  Shape: {df.count()} rows x {len(df.columns)} cols")
    print(f"  Columns: {df.columns[:5]}...")

    df.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"`workspace`.`ds-raw-datasets`.`{table_name}`")
