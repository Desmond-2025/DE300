#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
assignment4_airflow_dag.py

This DAG orchestrates your assignment tasks on AWS using Airflow.
It performs the following operations:
  1. Installs required dependencies.
  2. Loads source data (heart_disease.csv) from S3.
  3. Distributes the workflow into three branches:
      - scikit‑learn based EDA/modeling (Assignment 1).
      - Spark based EDA/modeling (Assignment 3).
      - Web scraping operations (Assignment 2).
  4. Optionally merges results.

The S3 functions (download/upload) are adapted from your wine quality pipeline.
"""

import os
import boto3
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
import subprocess
import logging

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator

# -----------------------------------------
# AWS S3 Configuration (update as needed)
# -----------------------------------------
S3_BUCKET = os.getenv("desmond-hw4-airflow", "desmond-hw4-airflow")
S3_PREFIX = ""  # update if files are under a specific prefix

# AWS Credentials (update with your valid credentials)
MY_KEY_ID = 'ASIAYAAO5HRMPV7YXSC2'
MY_ACCESS_KEY = 'pW2ILXz0ZR5/4Vc4HULcA4y6VpSJESA6pVHCwggp'
MY_SESSION_TOKEN = ('IQoJb3JpZ2luX2VjEDoaCXVzLWVhc3QtMiJHMEUCIQCPEeno+c+lzWz/pBG0R5Ax3UCBnF7ADuVGf9VlmWJNxAIgScUrlTZO/gZr8Kbf7+YJfaQc5EGK1RIt3BizKPGvjJAq9AIIg///////////ARAAGgw1NDk3ODcwOTAwMDgiDNpLJTX2wOMKF3pAtSrIAnPUf00l6hZrFo/N4UeBw6YTNgs2/9K3LbcieY3Ndo2iKCN+4VCUh+8UtdnKx1wfwL4Y33+t0Vc0mkklQGd09yvqhCzb3OLDVQ+vIVQHgCZe8eXyM5F4MmXhkf95iryRN8u4v5oZtViDpi9xRP5eXprksQrXMIVkPzGfGj7YL+5/zmfBGsSVjcT56FrLUEJlRX/esQw9jwSx0OCSXuhxS1L28TzeYNrY5B0DTJ+JO0VA28AiBf56DS2YAqTT2EHk9JvSSwY1bYMr15U7eYul2Rl/cMQkitfDW1siTXoYTYSqYfZV9XUr/LSBQa3oQJCPk7h7v/w28t7NQWs66zqqN4tPLLAOEHdKrwNlNeOn7KQHHOynbX5h0T7swoM9wpV0/FvYwYz1dqqwEBt8con+lU3Oibx4UgqCQZyShBsOQvbmX/Z0sOJckuIw04q5vgY6pwFVVE8rjBQ4KwoN5ZrzE6HzyDyTMx/E71dgHTdZaSpb/1cqlb76LJGBMGfCcpiQ8YOV4x7ud3w8SkFR+mO5Kl/lFcue+Wzi4IRultur9NaZ2spoPpGWr++kzUG2un4Dc3uqc31UeRHTMsN8Jutk112wtihJ0EItZUScF0Xiv5eMJxH0aWO858mkH4tNnTTNVqoE+6BQBsH5/4enpibQ/89500vJrFS19Q==')

s3_client = boto3.client('s3',
                         aws_access_key_id=MY_KEY_ID,
                         aws_secret_access_key=MY_ACCESS_KEY,
                         aws_session_token=MY_SESSION_TOKEN)

def upload_to_s3(data: pd.DataFrame, filename: str):
    """Upload a DataFrame as CSV to S3."""
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=S3_BUCKET, Key=f"{S3_PREFIX}/{filename}", Body=csv_buffer.getvalue())
    logging.info("Uploaded %s to S3 bucket %s", filename, S3_BUCKET)

def download_from_s3(filename: str) -> pd.DataFrame:
    """Download a CSV file from S3 into a DataFrame."""
    obj = s3_client.get_object(Bucket=S3_BUCKET, Key=f"{S3_PREFIX}/{filename}")
    df = pd.read_csv(obj['Body'])
    logging.info("Downloaded %s from S3 bucket %s", filename, S3_BUCKET)
    return df

# -----------------------------------------
# Default DAG Arguments
# -----------------------------------------
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 1),
    'retries': 1,
}

# Define the DAG
dag = DAG(
    'assignment4_airflow',
    default_args=default_args,
    schedule_interval='@once',
    catchup=False,
)

# -----------------------------------------
# Task: Install Dependencies
# -----------------------------------------
install_dependencies = BashOperator(
    task_id="install_dependencies",
    bash_command="pip install pandas boto3 scikit-learn pyspark requests",
    dag=dag,
)

# -----------------------------------------
# Task: Load Data from S3 (Common for all branches)
# -----------------------------------------
def load_data():
    """
    Download the common source data (heart_disease.csv) from S3 and re-upload it as 'data.csv'
    for subsequent tasks. This assumes heart_disease.csv is already in your S3 bucket.
    """
    try:
        df = download_from_s3("heart_disease.csv")
        # (Optional) You could perform minimal processing here, e.g. logging the data shape.
        logging.info("heart_disease.csv shape: %s", df.shape)
        # Re-upload as data.csv for consistency.
        upload_to_s3(df, "data.csv")
        logging.info("Common data loaded and re-uploaded as data.csv.")
    except Exception as e:
        logging.error("Failed to load data: %s", e)
        raise e

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

# -----------------------------------------
# Branch 1: EDA/Modeling using scikit-learn (Assignment 1)
# -----------------------------------------
def eda_sklearn():
    """
    Run scikit-learn based EDA, feature engineering (FE-1), and modeling.
    This reuses code from your assignment_1.ipynb (converted to assignment1.py).
    """
    logging.info("Starting scikit-learn EDA and modeling (Assignment 1)...")
    script_path = '/usr/local/airflow/dags/assignment1.py'

    if not os.path.exists(script_path):
        logging.error("Script not found: %s", script_path)
        return
    try:
        subprocess.run(["python", script_path], check=True)
        logging.info("scikit-learn branch completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error("Error in scikit-learn branch: %s", e)
        raise e

eda_sklearn_task = PythonOperator(
    task_id='eda_sklearn',
    python_callable=eda_sklearn,
    dag=dag
)

# -----------------------------------------
# Branch 2: EDA/Modeling using Spark (Assignment 3)
# -----------------------------------------
def eda_spark():
    """
    Execute Spark-based EDA, feature engineering (FE-2), and modeling.
    This reuses your code from assignment3.py.
    """
    logging.info("Starting Spark-based EDA and modeling (Assignment 3)...")
    script_path = '/usr/local/airflow/dags/assignment3.py'
    if not os.path.exists(script_path):
        logging.error("Script not found: %s", script_path)
        return
    try:
        subprocess.run(["python", script_path], check=True)
        logging.info("Spark branch completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error("Error in Spark branch: %s", e)
        raise e

eda_spark_task = PythonOperator(
    task_id='eda_spark',
    python_callable=eda_spark,
    dag=dag
)

# -----------------------------------------
# Branch 3: Web Scraping Operations (Assignment 2)
# -----------------------------------------
def web_scraping():
    """
    Run web scraping operations by executing the code from your assignment2 (1).ipynb,
    which has been converted to assignment2.py.
    """
    logging.info("Starting web scraping operations (Assignment 2)...")
    script_path = '/usr/local/airflow/dags/assignment2.py'
    if not os.path.exists(script_path):
        logging.error("Script not found: %s", script_path)
        return
    try:
        subprocess.run(["python", script_path], check=True)
        logging.info("Web scraping branch completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error("Error in web scraping branch: %s", e)
        raise e

web_scraping_task = PythonOperator(
    task_id='web_scraping',
    python_callable=web_scraping,
    dag=dag
)

# -----------------------------------------
# Optional Join Task: Aggregate/Finalize Workflow
# -----------------------------------------
join_task = DummyOperator(
    task_id='join_tasks',
    trigger_rule='all_done',
    dag=dag,
)

# -----------------------------------------
# Define DAG Dependencies
# -----------------------------------------
# The dependencies are set as follows:
# install_dependencies >> load_data >> (eda_sklearn, eda_spark, web_scraping) >> join_task
install_dependencies >> load_data_task
load_data_task >> [eda_sklearn_task, eda_spark_task, web_scraping_task] >> join_task
