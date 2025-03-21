#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Tue Mar  4 11:35:39 2025
Updated for Wine Quality Data using S3 where wine.csv is stored in S3.
"""

import boto3
import os
import pandas as pd
from io import StringIO
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

# AWS S3 Configuration (update as needed)
S3_BUCKET = os.getenv("desmond-mwaa-lab8 ", "desmond-mwaa-lab8 ")
S3_PREFIX = ""  

# credentials 
MY_KEY_ID = 'ASIAYAAO5HRMPV7YXSC2'
MY_ACCESS_KEY = 'pW2ILXz0ZR5/4Vc4HULcA4y6VpSJESA6pVHCwggp'
MY_SESSION_TOKEN = 'IQoJb3JpZ2luX2VjEDoaCXVzLWVhc3QtMiJHMEUCIQCPEeno+c+lzWz/pBG0R5Ax3UCBnF7ADuVGf9VlmWJNxAIgScUrlTZO/gZr8Kbf7+YJfaQc5EGK1RIt3BizKPGvjJAq9AIIg///////////ARAAGgw1NDk3ODcwOTAwMDgiDNpLJTX2wOMKF3pAtSrIAnPUf00l6hZrFo/N4UeBw6YTNgs2/9K3LbcieY3Ndo2iKCN+4VCUh+8UtdnKx1wfwL4Y33+t0Vc0mkklQGd09yvqhCzb3OLDVQ+vIVQHgCZe8eXyM5F4MmXhkf95iryRN8u4v5oZtViDpi9xRP5eXprksQrXMIVkPzGfGj7YL+5/zmfBGsSVjcT56FrLUEJlRX/esQw9jwSx0OCSXuhxS1L28TzeYNrY5B0DTJ+JO0VA28AiBf56DS2YAqTT2EHk9JvSSwY1bYMr15U7eYul2Rl/cMQkitfDW1siTXoYTYSqYfZV9XUr/LSBQa3oQJCPk7h7v/w28t7NQWs66zqqN4tPLLAOEHdKrwNlNeOn7KQHHOynbX5h0T7swoM9wpV0/FvYwYz1dqqwEBt8con+lU3Oibx4UgqCQZyShBsOQvbmX/Z0sOJckuIw04q5vgY6pwFVVE8rjBQ4KwoN5ZrzE6HzyDyTMx/E71dgHTdZaSpb/1cqlb76LJGBMGfCcpiQ8YOV4x7ud3w8SkFR+mO5Kl/lFcue+Wzi4IRultur9NaZ2spoPpGWr++kzUG2un4Dc3uqc31UeRHTMsN8Jutk112wtihJ0EItZUScF0Xiv5eMJxH0aWO858mkH4tNnTTNVqoE+6BQBsH5/4enpibQ/89500vJrFS19Q=='

s3_client = boto3.client('s3',
                         aws_access_key_id=MY_KEY_ID,
                         aws_secret_access_key=MY_ACCESS_KEY,
                         aws_session_token=MY_SESSION_TOKEN)

def upload_to_s3(data, filename):
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=S3_BUCKET, Key=f"{S3_PREFIX}/{filename}", Body=csv_buffer.getvalue())

def download_from_s3(filename):
    obj = s3_client.get_object(Bucket=S3_BUCKET, Key=f"{S3_PREFIX}/{filename}")
    return pd.read_csv(obj['Body'])

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 1),
    'retries': 1,
}

dag = DAG(
    'ml_pipeline_s3',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
)

# Install dependencies
install_dependencies = BashOperator(
    task_id="install_dependencies",
    bash_command="pip install pandas boto3 scikit-learn",
    dag=dag,
)

def generate_data():
    # Download wine.csv from S3 (wine.csv should be stored under S3_PREFIX)
    df = download_from_s3("wine.csv")
    # Create binary label: 1 if quality >= 6, else 0
    if 'quality' in df.columns and 'label' not in df.columns:
         df['label'] = (df['quality'] >= 6).astype(int)
    # Upload the processed data as 'data.csv' for subsequent tasks
    upload_to_s3(df, "data.csv")

generate_task = PythonOperator(
    task_id="generate_data",
    python_callable=generate_data,
    dag=dag,
)

def clean_impute():
    df = download_from_s3("data.csv")
    df.fillna(df.mean(), inplace=True)
    upload_to_s3(df, "cleaned_data.csv")

clean_task = PythonOperator(
    task_id="clean_impute",
    python_callable=clean_impute,
    dag=dag,
)

def feature_engineering():
    df = download_from_s3("cleaned_data.csv")
    # Create a new feature using the 'alcohol' column
    df['new_feature'] = df['alcohol'] * 0.5
    upload_to_s3(df, "fe_data.csv")

feature_task = PythonOperator(
    task_id="feature_engineering",
    python_callable=feature_engineering,
    dag=dag,
)

def train_svm():
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    df = download_from_s3("fe_data.csv")
    X = df[['alcohol', 'new_feature']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = SVC()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"SVM Accuracy: {acc}")

svm_task = PythonOperator(
    task_id="train_svm",
    python_callable=train_svm,
    dag=dag,
)

def train_logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    df = download_from_s3("fe_data.csv")
    X = df[['alcohol', 'new_feature']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Logistic Regression Accuracy: {acc}")

logistic_task = PythonOperator(
    task_id="train_logistic",
    python_callable=train_logistic,
    dag=dag,
)

def merge_results():
    print("Merging model results and selecting the best model.")

merge_task = PythonOperator(
    task_id="merge_results",
    python_callable=merge_results,
    dag=dag,
)

def evaluate_test():
    print("Evaluating final model performance.")

evaluate_task = PythonOperator(
    task_id="evaluate_test",
    python_callable=evaluate_test,
    dag=dag,
)

# Set DAG dependencies
install_dependencies >> generate_task
generate_task >> clean_task
clean_task >> feature_task
feature_task >> [svm_task, logistic_task]
[svm_task, logistic_task] >> merge_task
merge_task >> evaluate_task
