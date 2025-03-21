#!/usr/bin/env python
# coding: utf-8


# Loading required packages

import boto3
from io import BytesIO
import pandas as pd
# you need to change the credentials for yourself
# Note that aws_access_key_id changes from time to time

s3 = boto3.client('s3',
                  aws_access_key_id='ASIAYAAO5HRMPV7YXSC2',
                  aws_secret_access_key='pW2ILXz0ZR5/4Vc4HULcA4y6VpSJESA6pVHCwggp',
                  aws_session_token='IQoJb3JpZ2luX2VjEDoaCXVzLWVhc3QtMiJHMEUCIQCPEeno+c+lzWz/pBG0R5Ax3UCBnF7ADuVGf9VlmWJNxAIgScUrlTZO/gZr8Kbf7+YJfaQc5EGK1RIt3BizKPGvjJAq9AIIg///////////ARAAGgw1NDk3ODcwOTAwMDgiDNpLJTX2wOMKF3pAtSrIAnPUf00l6hZrFo/N4UeBw6YTNgs2/9K3LbcieY3Ndo2iKCN+4VCUh+8UtdnKx1wfwL4Y33+t0Vc0mkklQGd09yvqhCzb3OLDVQ+vIVQHgCZe8eXyM5F4MmXhkf95iryRN8u4v5oZtViDpi9xRP5eXprksQrXMIVkPzGfGj7YL+5/zmfBGsSVjcT56FrLUEJlRX/esQw9jwSx0OCSXuhxS1L28TzeYNrY5B0DTJ+JO0VA28AiBf56DS2YAqTT2EHk9JvSSwY1bYMr15U7eYul2Rl/cMQkitfDW1siTXoYTYSqYfZV9XUr/LSBQa3oQJCPk7h7v/w28t7NQWs66zqqN4tPLLAOEHdKrwNlNeOn7KQHHOynbX5h0T7swoM9wpV0/FvYwYz1dqqwEBt8con+lU3Oibx4UgqCQZyShBsOQvbmX/Z0sOJckuIw04q5vgY6pwFVVE8rjBQ4KwoN5ZrzE6HzyDyTMx/E71dgHTdZaSpb/1cqlb76LJGBMGfCcpiQ8YOV4x7ud3w8SkFR+mO5Kl/lFcue+Wzi4IRultur9NaZ2spoPpGWr++kzUG2un4Dc3uqc31UeRHTMsN8Jutk112wtihJ0EItZUScF0Xiv5eMJxH0aWO858mkH4tNnTTNVqoE+6BQBsH5/4enpibQ/89500vJrFS19Q==')

bucket_name = 'desmond-hw4-airflow'
object_key = 'heart_disease.csv'
csv_obj = s3.get_object(Bucket=bucket_name, Key=object_key)
body = csv_obj['Body']
csv_string = body.read().decode('utf-8')
df = pd.read_csv(BytesIO(csv_string.encode()))
print(df.head())

# Cleaning the Data

# Cleaning Step 1
# List of columns to retain
columns_to_retain = [
    'age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 'fbs',
    'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak', 'slope']

# Retaining the specified columns
df_cleaned = df[columns_to_retain]

print(df_cleaned.head())

# Cleaning Step 2

# Creating a copy of the dataframe to avoid SettingWithCopyWarning
df_cleaned = df[columns_to_retain].copy()

# Step 1: Impute missing values for 'painloc' and 'painexer'
df_cleaned.loc[:, 'painloc'] = df_cleaned['painloc'].fillna(df_cleaned['painloc'].mode()[0])
df_cleaned.loc[:, 'painexer'] = df_cleaned['painexer'].fillna(df_cleaned['painexer'].mode()[0])

# Step 2: Replace values less than 100 mm Hg for 'trestbps'
df_cleaned.loc[:, 'trestbps'] = df_cleaned['trestbps'].apply(lambda x: x if x >= 100 else df_cleaned['trestbps'].median())

# Step 3: Replace values less than 0 and greater than 4 for 'oldpeak'
df_cleaned.loc[:, 'oldpeak'] = df_cleaned['oldpeak'].apply(lambda x: x if 0 <= x <= 4 else df_cleaned['oldpeak'].median())

# Step 4: Impute missing values for 'thaldur' and 'thalach'
df_cleaned.loc[:, 'thaldur'] = df_cleaned['thaldur'].fillna(df_cleaned['thaldur'].median())
df_cleaned.loc[:, 'thalach'] = df_cleaned['thalach'].fillna(df_cleaned['thalach'].median())

# Step 5: Replace missing values and values greater than 1 for 'fbs', 'prop', 'nitr', 'pro', 'diuretic'
for col in ['fbs', 'prop', 'nitr', 'pro', 'diuretic']:
    df_cleaned.loc[:, col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
    df_cleaned.loc[:, col] = df_cleaned[col].apply(lambda x: 1 if x > 1 else x)

# Step 6: Impute missing values for 'exang' and 'slope'
df_cleaned.loc[:, 'exang'] = df_cleaned['exang'].fillna(df_cleaned['exang'].mode()[0])
df_cleaned.loc[:, 'slope'] = df_cleaned['slope'].fillna(df_cleaned['slope'].mode()[0])

print(df_cleaned.head())

# Cleaning Step 3
# For Source 1, I used the "Proportion of people 15 years and over who were current daily smokers by age and sex, 2022" dataset 
# to impute smoking rates more precisely by considering both age and sex.
# For Source 2, I used the tobacco product use - cigarettes dataset assuming that smoking rates by different age groups for both men and women are the same.

# Updated smoking rates for Source 1 and Source 2
smoking_rates_source1 = {
    '15–17': {'Male': 1.2, 'Female': 1.8},
    '18–24': {'Male': 9.3, 'Female': 5.9},
    '25–34': {'Male': 13.4, 'Female': 8.8},
    '35–44': {'Male': 13.5, 'Female': 8.5},
    '45–54': {'Male': 15.3, 'Female': 11.6},
    '55–64': {'Male': 17.4, 'Female': 12.0},
    '65–74': {'Male': 9.9, 'Female': 7.9},
    '75+': {'Male': 3.8, 'Female': 1.9}
}

smoking_rates_source2 = {
    '18–24': {'Male': 4.8, 'Female': 4.8},
    '25–44': {'Male': 12.5, 'Female': 12.5},
    '45–64': {'Male': 15.1, 'Female': 15.1},
    '65+': {'Male': 8.7, 'Female': 8.7}
}

source2_smoking_rate_among_men = 13.2
source2_smoking_rate_among_women = 10.0

# Convert the 'age' column to numeric, coercing any non-numeric values to NaN
df_cleaned['age'] = pd.to_numeric(df_cleaned['age'], errors='coerce')

# For Source 1, simply replace missing values with the corresponding smoking rates for age groups
def impute_smoking_rate_source1(row):
    age_group = None
    if 15 <= row['age'] <= 17:
        age_group = '15–17'
    elif 18.0 <= row['age'] <= 24.0:
        age_group = '18–24'
    elif 25.0 <= row['age'] <= 34.0:
        age_group = '25–34'
    elif 35.0 <= row['age'] <= 44.0:
        age_group = '35–44'
    elif 45.0 <= row['age'] <= 54.0:
        age_group = '45–54'
    elif 55.0 <= row['age'] <= 64.0:
        age_group = '55–64'
    elif 65 <= row['age'] <= 74:
        age_group = '65–74'
    elif row['age'] >= 75:
        age_group = '75+'
    if age_group is None:
        return 0
    if row['sex'] == '0':  # Female
        return smoking_rates_source1[age_group]['Female']
    else:  # Male
        return smoking_rates_source1[age_group]['Male']

# For Source 2, apply the formula for males
def impute_smoking_rate_source2(row):
    age_group = None
    if 18 <= row['age'] <= 24:
        age_group = '18–24'
    elif 25 <= row['age'] <= 44:
        age_group = '25–44'
    elif 45 <= row['age'] <= 64:
        age_group = '45–64'
    elif row['age'] >= 65:
        age_group = '65+'
    if age_group is None:
        return 0
    if row['sex'] == '0':  # Female
        return smoking_rates_source2[age_group]['Female']
    else:  # Male
        male_rate = smoking_rates_source2[age_group]['Male']
        female_rate = smoking_rates_source2[age_group]['Female']
        return male_rate * (source2_smoking_rate_among_men / source2_smoking_rate_among_women)

# Apply the imputation for Source 1 and Source 2
df_cleaned['smoke_imputed_source1'] = df_cleaned.apply(impute_smoking_rate_source1, axis=1)
df_cleaned['smoke_imputed_source2'] = df_cleaned.apply(impute_smoking_rate_source2, axis=1)

# Normalize the imputed columns
df_cleaned['smoke_imputed_source1_normalized'] = (df_cleaned['smoke_imputed_source1'] - df_cleaned['smoke_imputed_source1'].min()) / (df_cleaned['smoke_imputed_source1'].max() - df_cleaned['smoke_imputed_source1'].min())
df_cleaned['smoke_imputed_source2_normalized'] = (df_cleaned['smoke_imputed_source2'] - df_cleaned['smoke_imputed_source2'].min()) / (df_cleaned['smoke_imputed_source2'].max() - df_cleaned['smoke_imputed_source2'].min())

print(df_cleaned.head())


from sklearn.model_selection import train_test_split

# Task 3
# Features (X) are the columns in df_cleaned
X = df_cleaned  # Features

# Target (y) is the 'target' column from the original df
y = df['target']  # Target labels

# Drop rows where the target (y) is NaN
X = X[y.notna()]  # Keep only the rows where y is not NaN
y = y.dropna()  # Remove NaN values from y

# Perform stratified split (90-10 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Impute missing values using the median
imputer = SimpleImputer(strategy='median') 
X_imputed = imputer.fit_transform(X)  # Apply imputation to X

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.1, stratify=y, random_state=42)

# Task 4

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Scale training data
X_test_scaled = scaler.transform(X_test)  # Scale test data

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

# Hyperparameter tuning
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
}
param_grid_lr = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}

# 5-fold cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# For each model, perform 5-fold cross-validation and hyperparameter tuning
results = {}
for model_name, model in models.items():
    print(f"Training {model_name}")
    
    if model_name == 'Random Forest':
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid_rf, cv=cv, scoring='accuracy')
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
    elif model_name == 'Logistic Regression':
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid_lr, cv=cv, scoring='accuracy')
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model  # No hyperparameter tuning for SVM and KNN for simplicity
        
    # 5-fold cross-validation
    cv_results = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    results[model_name] = cv_results

# Reporting performance
for model_name, cv_results in results.items():
    print(f"{model_name} cross-validation results:")
    print(f"Accuracy: Mean={cv_results.mean():.4f}, Std={cv_results.std():.4f}")
    
    # Train on the full dataset and assess performance
    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)
    
    # Print classification report (Precision, Recall, F1-score)
    print(classification_report(y_test, y_pred))
    
    # Calculate and print ROC AUC
    if hasattr(best_model, "predict_proba"):
        roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
    else:
        roc_auc = "Not available"
    print(f"ROC AUC: {roc_auc}\n")

# Key Performance Metrics:
# - Accuracy: The proportion of correctly predicted labels.
# - Precision: The proportion of positive predictions that were correct.
# - Recall: The proportion of actual positives correctly identified.
# - F1-score: The harmonic mean of precision and recall.
# - ROC AUC: Provides an aggregate measure of performance across classification thresholds.

# Summary of Results:
# - Logistic Regression, Random Forest, SVM, and KNN show similar cross-validation accuracy (~80%).
# - Precision, recall, and F1-score are consistent across models.
# - ROC AUC of ~0.795 indicates a good model performance for distinguishing classes.
#
# Recommendations:
# - Logistic Regression and Random Forest perform similarly; choose based on interpretability or speed.
# - SVM and KNN perform slightly lower in accuracy.
# - Random Forest is recommended for its balance between performance and interpretability, especially for heart disease prediction.

# End of script.
