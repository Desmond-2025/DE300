#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import boto3
from io import BytesIO

s3 = boto3.client('s3',
                  aws_access_key_id='ASIAYAAO5HRMPV7YXSC2',
                  aws_secret_access_key='pW2ILXz0ZR5/4Vc4HULcA4y6VpSJESA6pVHCwggp',
                  aws_session_token='IQoJb3JpZ2luX2VjEDoaCXVzLWVhc3QtMiJHMEUCIQCPEeno+c+lzWz/pBG0R5Ax3UCBnF7ADuVGf9VlmWJNxAIgScUrlTZO/gZr8Kbf7+YJfaQc5EGK1RIt3BizKPGvjJAq9AIIg///////////ARAAGgw1NDk3ODcwOTAwMDgiDNpLJTX2wOMKF3pAtSrIAnPUf00l6hZrFo/N4UeBw6YTNgs2/9K3LbcieY3Ndo2iKCN+4VCUh+8UtdnKx1wfwL4Y33+t0Vc0mkklQGd09yvqhCzb3OLDVQ+vIVQHgCZe8eXyM5F4MmXhkf95iryRN8u4v5oZtViDpi9xRP5eXprksQrXMIVkPzGfGj7YL+5/zmfBGsSVjcT56FrLUEJlRX/esQw9jwSx0OCSXuhxS1L28TzeYNrY5B0DTJ+JO0VA28AiBf56DS2YAqTT2EHk9JvSSwY1bYMr15U7eYul2Rl/cMQkitfDW1siTXoYTYSqYfZV9XUr/LSBQa3oQJCPk7h7v/w28t7NQWs66zqqN4tPLLAOEHdKrwNlNeOn7KQHHOynbX5h0T7swoM9wpV0/FvYwYz1dqqwEBt8con+lU3Oibx4UgqCQZyShBsOQvbmX/Z0sOJckuIw04q5vgY6pwFVVE8rjBQ4KwoN5ZrzE6HzyDyTMx/E71dgHTdZaSpb/1cqlb76LJGBMGfCcpiQ8YOV4x7ud3w8SkFR+mO5Kl/lFcue+Wzi4IRultur9NaZ2spoPpGWr++kzUG2un4Dc3uqc31UeRHTMsN8Jutk112wtihJ0EItZUScF0Xiv5eMJxH0aWO858mkH4tNnTTNVqoE+6BQBsH5/4enpibQ/89500vJrFS19Q==')

bucket_name = 'desmond-hw4-airflow'
object_key = 'data.csv'
csv_obj = s3.get_object(Bucket=bucket_name, Key=object_key)
body = csv_obj['Body']
csv_string = body.read().decode('utf-8')
data = pd.read_csv(BytesIO(csv_string.encode()))
print(data.head())

# MISSING VALUES
# checking for missing values
missing_values = data.isnull().sum()
missing_percentage = (missing_values / len(data)) * 100

# Create a DataFrame to display missing values and their percentage
missing_data = pd.DataFrame({"Missing Values": missing_values, "Percentage": missing_percentage})
missing_data = missing_data[missing_data["Missing Values"] > 0].sort_values(by="Percentage", ascending=False)
print(missing_data)

# IMPUTATION

# Impute missing values based on the chosen strategies
# Categorical columns (mode imputation)
data['TOTALAREA_MODE'].fillna(data['TOTALAREA_MODE'].mode()[0], inplace=True)
data['HOUSETYPE_MODE'].fillna(data['HOUSETYPE_MODE'].mode()[0], inplace=True)

# Numerical columns (median imputation)
data['EXT_SOURCE_1'].fillna(data['EXT_SOURCE_1'].median(), inplace=True)
data['EXT_SOURCE_2'].fillna(data['EXT_SOURCE_2'].median(), inplace=True)
data['EXT_SOURCE_3'].fillna(data['EXT_SOURCE_3'].median(), inplace=True)

# Impute missing values for AMT_REQ_CREDIT_BUREAU_YEAR with zero
data['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0, inplace=True)

# Verify if the missing values were imputed successfully
missing_values_after_imputation = data.isnull().sum()

# HANDLING OUTLIERS

# Identify numerical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns

# Define capping threshold for the 99th and 1st percentiles
percentile_99 = data[numerical_cols].quantile(0.99)
percentile_1 = data[numerical_cols].quantile(0.01)

# Calculate IQR for each numerical feature
Q1 = data[numerical_cols].quantile(0.25)
Q3 = data[numerical_cols].quantile(0.75)
IQR = Q3 - Q1

# Capping the values at the 1st and 99th percentiles for appropriate columns
for col in numerical_cols:
    if col not in ['DAYS_BIRTH']:  # Skip DAYS_BIRTH since it has no outliers
        data[col] = data[col].clip(lower=percentile_1[col], upper=percentile_99[col])

# For DAYS_EMPLOYED, replace extreme negative values with the median
data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED'].apply(lambda x: x if x >= 0 else data['DAYS_EMPLOYED'].median())

# For CNT_CHILDREN, CNT_FAM_MEMBERS, cap at reasonable values (e.g., 10)
data['CNT_CHILDREN'] = data['CNT_CHILDREN'].clip(upper=10)
data['CNT_FAM_MEMBERS'] = data['CNT_FAM_MEMBERS'].clip(upper=10)


# COMPUTING STATISTIC MEASURES
statistics = data[numerical_cols].describe().T
statistics['Skewness'] = data[numerical_cols].skew()
statistics['Kurtosis'] = data[numerical_cols].kurtosis()

print(statistics)


# Analysis of Statistics Measures:
# TARGET: Right-skewed distribution and heavy tails indicate most clients have no payment difficulties with some extreme cases.
# CNT_CHILDREN: Moderately right-skewed, concentrated toward fewer children.
# CNT_FAM_MEMBERS: Mild right skew, mostly 2 to 3 members.
# AMT_INCOME_TOTAL: Positively skewed, indicating lower incomes for most.
# AMT_CREDIT: Right-skewed with a few high credit amounts.
# TOTALAREA_MODE: Right-skewed with heavy tails.
# EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3: Nearly symmetric with moderate peaks.
# DAYS_EMPLOYED: Right-skewed, with many negative values replaced.
# AMT_REQ_CREDIT_BUREAU_YEAR: Fairly symmetric with a slight skew.


data[data['DAYS_BIRTH'] > 0]


# FEATURE TRANSFORMATIONS

# Separate numerical and categorical features
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# 1. Numerical features transformation:
# Apply Standard Scaling to numerical features (except for those to be log-transformed)
scaler = StandardScaler()

# List of features to apply log transformation
log_transform_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT']

# Apply log transformation to skewed numerical features
for col in log_transform_features:
    data[col] = np.log1p(data[col])  # Log transform (log(x+1))

# Standard scale the rest of the numerical features
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# For DAYS_EMPLOYED, convert negative values to binary (1 for employed, 0 for unemployed)
data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED'].apply(lambda x: 1 if x >= 0 else 0)

# 2. Categorical features transformation:
# Label encoding for binary categorical features like 'CODE_GENDER', 'FLAG_OWN_CAR', etc.
binary_categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_HOUSING_TYPE']

encoder = LabelEncoder()

for col in binary_categorical_cols:
    data[col] = encoder.fit_transform(data[col])

# One-Hot Encoding for non-binary categorical features
data = pd.get_dummies(data, columns=[col for col in categorical_cols if col not in binary_categorical_cols])

print(data)


# PLOTS

# Plotting box plots for numerical features
plt.figure(figsize=(20, 15))

# Box plots for key numerical features
numerical_features_for_boxplot = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_EMPLOYED', 'TOTALAREA_MODE']

for i, feature in enumerate(numerical_features_for_boxplot, 1):
    plt.subplot(2, 2, i)
    plt.boxplot(data[feature])
    plt.title(f'Boxplot of {feature}')
    plt.tight_layout()

plt.show()


# Plotting scatter plots for numerical features
plt.figure(figsize=(15, 10))

# Scatter plots for pairwise numerical features
pairs_to_plot = [('AMT_INCOME_TOTAL', 'AMT_CREDIT'), ('AMT_INCOME_TOTAL', 'TOTALAREA_MODE'), 
                 ('AMT_CREDIT', 'TOTALAREA_MODE'), ('DAYS_EMPLOYED', 'TOTALAREA_MODE')]

for i, (x, y) in enumerate(pairs_to_plot, 1):
    plt.subplot(2, 2, i)
    plt.scatter(data[x], data[y], alpha=0.5)
    plt.title(f'Scatter plot: {x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()

plt.show()

# PLOT ANALYSIS:
# Scatter Plot Analysis:
# - AMT_INCOME_TOTAL vs AMT_CREDIT: Positive correlation with some outliers.
# - AMT_INCOME_TOTAL vs TOTALAREA_MODE: No clear pattern.
# - AMT_CREDIT vs TOTALAREA_MODE: No strong relationship.
# - DAYS_EMPLOYED vs TOTALAREA_MODE: No significant pattern.
#
# Box Plot Analysis:
# - AMT_INCOME_TOTAL: Wide spread with potential outliers.
# - AMT_CREDIT: Wide range with extreme values.
# - DAYS_EMPLOYED: Concentration near zero indicating many non-employed clients.


# Save the cleaned and transformed DataFrame to a new CSV file
data.to_csv('cleaned_transformed_data.csv', index=False)
print("Data has been saved to 'cleaned_transformed_data.csv' successfully!")


print(data)
