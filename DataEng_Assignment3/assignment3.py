from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer
from pyspark.sql.functions import col, when, lit, udf, monotonically_increasing_id, isnan
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# SOME NOTES
'''
- Because I had issues running spark on my machine and using the docker image, the professor suggested I 
write a python file and run spark locally on my machine. I manually included the outputs from the spark console
- That also meant I could not directly connect to my S3 bucket so I wrote the code for it and commented out the 
call that reads from the csv. I substituted it with a read call to the csv file that I downloaded to the same directory

'''

# ----- LOADING -----------------------------------------------------------------
# Create a SparkSession
spark = SparkSession.builder \
    .appName("HeartDiseaseData") \
    .getOrCreate()

# Set S3 credentials in the Hadoop configuration
hadoop_conf = spark._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.access.key", "ASIAYAAO5HRMONOOOIKC")
hadoop_conf.set("fs.s3a.secret.key", "7lzUKZaNI7apuhBglmwYkRgrXQK6kjEnZORPk5Vw")
hadoop_conf.set("fs.s3a.session.token", "IQoJb3JpZ2luX2VjEPz//////////wEaCXVzLWVhc3QtMiJIMEYCIQDYH1gxYEFocnTaHMxTjqHIZxhY6QiKTHCDZs5Zq7NT8QIhAPntSAdpMpWscrufsrq0wK/iTSGrDqEujjvTHHhVZTriKusCCDUQABoMNTQ5Nzg3MDkwMDA4IgwUebCzD+7lWRgUKvcqyAIfV4BQ31WaqB9GhrU757KxoqpY1sirv5tdtVxn1JEfcRE3Zkz0DvihrXq1tYE8URm0Ti1K3BXDl9KKjgqOuKjUURnTJXALM5MBr5G1xlOAuAhqqbjXtH5QkwwRTNDZMAlfuRHQSHMaXAp48YOWG0cVbiNtEzEo7F+at2hlB5sHQhtsx0x3lXLOjeBbPQXmo2XS15t7pt1cEnDXZLWxVknxyiNzZ4mtT5j8eNYNftK2YJZlxy7syJ07uT4Xk5MRGji7Fn2vD41T3vIf5lClraPzBCZ46IgpEU3XFCsQwM2hg6HA1n/Ib8pysKvWyTi5hEG1mrANPzGzzmrCxLQlRf5mrQ31px0UDGIu096sCsCZijRe+yPkiAOaj83NDflpVL6YVnQKKKikID1n9C/2ZUSmZCapM/TWYn8WqBNiWrCOghTnxSefTxeaMICj870GOqYB571YYinaDRlTPOuNjvZog8+amH9TAS/cye8A8J0AibZNiWNvcJG5ZzBnCIxPRm09Dnr7vaEeuuG0fvLbQ+R2e4aLTPoRkcGUu3DlWGijUxtfNAPrIaywPCW2T+RIRvvPqW2CFi6yViHjeWfel1HSe3PCxWjzD671Q2ESKyAAI5ZedPRvm+mMNxI9IgmFjvJeH9yDfuGcplmscYtugihwwjRgUn+Nkg==")

# Define the S3 bucket and file path
bucket = "de300winter2025"
object_key = "desmond_nebah/heart_disease.csv"

# Read the CSV file directly from S3 using Spark's CSV reader
#df = spark.read.csv(f"s3a://{bucket}/{object_key}", header=True, inferSchema=True)

df = spark.read.csv("heart_disease.csv", header=True, inferSchema=True)
# Show the first 5 rows of the DataFrame
# df.show(5)

# OUTPUT IS BELOW
'''
>>> exec(open("assignment3.py").read())
+---+---+-------+--------+-------+-------+---+--------+---+----+-----+----+-----+---+----+-------+-------+-----+----------+-----+---+----+----+---+--------+-----+-------+--------+----+-------+--------+--------+--------+-----+--------+-----+-----+-------+-----+-----+------+---+-------+-------+------+------+------+------+----+-------+-------+-------+---+----+---+------+
|age|sex|painloc|painexer|relrest|pncaden| cp|trestbps|htn|chol|smoke|cigs|years|fbs|  dm|famhist|restecg|ekgmo|ekgday(day|ekgyr|dig|prop|nitr|pro|diuretic|proto|thaldur|thaltime| met|thalach|thalrest|tpeakbps|tpeakbpd|dummy|trestbpd|exang|xhypo|oldpeak|slope|rldv5|rldv5e| ca|restckm|exerckm|restef|restwm|exeref|exerwm|thal|thalsev|thalpul|earlobe|cmo|cday|cyr|target|
+---+---+-------+--------+-------+-------+---+--------+---+----+-----+----+-----+---+----+-------+-------+-----+----------+-----+---+----+----+---+--------+-----+-------+--------+----+-------+--------+--------+--------+-----+--------+-----+-----+-------+-----+-----+------+---+-------+-------+------+------+------+------+----+-------+-------+-------+---+----+---+------+
| 63|  1|   null|    null|   null|   null|  1|     145|  1| 233| null|  50|   20|  1|null|      1|      2|    2|         3|   81|  0|   0|   0|  0|       0|    1|   10.5|     6.0|13.0|    150|      60|     190|      90|  145|      85|    0|    0|    2.3|    3| null|   172|  0|   null|   null|  null|  null|  null|  null|   6|   null|   null|   null|  2|  16| 81|     0|
| 67|  1|   null|    null|   null|   null|  4|     160|  1| 286| null|  40|   40|  0|null|      1|      2|    3|         5|   81|  0|   1|   0|  0|       0|    1|    9.5|     6.0|13.0|    108|      64|     160|      90|  160|      90|    1|    0|    1.5|    2| null|   185|  3|   null|   null|  null|  null|  null|  null|   3|   null|   null|   null|  2|   5| 81|     1|
| 67|  1|   null|    null|   null|   null|  4|     120|  1| 229| null|  20|   35|  0|null|      1|      2|    2|        19|   81|  0|   1|   0|  0|       0|    1|    8.5|     6.0|10.0|    129|      78|     140|      80|  120|      80|    1|    0|    2.6|    2| null|   150|  2|   null|   null|  null|  null|  null|  null|   7|   null|   null|   null|  2|  20| 81|     1|
| 37|  1|   null|    null|   null|   null|  3|     130|  0| 250| null|   0|    0|  0|null|      1|      0|    2|        13|   81|  0|   1|   0|  0|       0|    1|   13.0|    13.0|17.0|    187|      84|     195|      68|  130|      78|    0|    0|    3.5|    3| null|   167|  0|   null|   null|  null|  null|  null|  null|   3|   null|   null|   null|  2|   4| 81|     0|
| 41|  0|   null|    null|   null|   null|  2|     130|  1| 204| null|   0|    0|  0|null|      1|      2|    2|         7|   81|  0|   0|   0|  0|       0|    1|    7.0|    null| 9.0|    172|      71|     160|      74|  130|      86|    0|    0|    1.4|    1| null|    40|  0|   null|   null|  null|  null|  null|  null|   3|   null|   null|   null|  2|  18| 81|     0|
+---+---+-------+--------+-------+-------+---+--------+---+----+-----+----+-----+---+----+-------+-------+-----+----------+-----+---+----+----+---+--------+-----+-------+--------+----+-------+--------+--------+--------+-----+--------+-----+-----+-------+-----+-----+------+---+-------+-------+------+------+------+------+----+-------+-------+-------+---+----+---+------+
only showing top 5 rows

'''

# ----- CLEANING -------------------------------------------------------------------

# Cleaning Step 1

# Define the list of columns to retain
columns_to_retain = [
    'age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 'fbs',
    'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak', 'slope'
]

# Retain the specified columns using Spark's select method
df_cleaned = df.select(*columns_to_retain)

# Show the first 5 rows of the cleaned DataFrame
# df_cleaned.show(5)

# OUTPUT IS BELOW

'''
>>> exec(open("assignment3.py").read())
+---+---+-------+--------+---+--------+-----+---+----+----+---+--------+-------+-------+-----+-------+-----+
|age|sex|painloc|painexer| cp|trestbps|smoke|fbs|prop|nitr|pro|diuretic|thaldur|thalach|exang|oldpeak|slope|
+---+---+-------+--------+---+--------+-----+---+----+----+---+--------+-------+-------+-----+-------+-----+
| 63|  1|   null|    null|  1|     145| null|  1|   0|   0|  0|       0|   10.5|    150|    0|    2.3|    3|
| 67|  1|   null|    null|  4|     160| null|  0|   1|   0|  0|       0|    9.5|    108|    1|    1.5|    2|
| 67|  1|   null|    null|  4|     120| null|  0|   1|   0|  0|       0|    8.5|    129|    1|    2.6|    2|
| 37|  1|   null|    null|  3|     130| null|  0|   1|   0|  0|       0|   13.0|    187|    0|    3.5|    3|
| 41|  0|   null|    null|  2|     130| null|  0|   0|   0|  0|       0|    7.0|    172|    0|    1.4|    1|
+---+---+-------+--------+---+--------+-----+---+----+----+---+--------+-------+-------+-----+-------+-----+
only showing top 5 rows

'''

# Cleaning Step 2

# ----- Step 1: Impute missing values for 'painloc' and 'painexer' using mode -----
# Compute mode for 'painloc'
painloc_mode_row = df_cleaned.filter(col("painloc").isNotNull()) \
    .groupBy("painloc").count().orderBy("count", ascending=False).first()
painloc_mode = painloc_mode_row["painloc"] if painloc_mode_row is not None else None

# Compute mode for 'painexer'
painexer_mode_row = df_cleaned.filter(col("painexer").isNotNull()) \
    .groupBy("painexer").count().orderBy("count", ascending=False).first()
painexer_mode = painexer_mode_row["painexer"] if painexer_mode_row is not None else None

# Build a fill dictionary only if we have non-null mode values
fill_dict = {}
if painloc_mode is not None:
    fill_dict["painloc"] = painloc_mode
if painexer_mode is not None:
    fill_dict["painexer"] = painexer_mode

if fill_dict:
    df_cleaned = df_cleaned.na.fill(fill_dict)


# ----- Step 2: Replace 'trestbps' values less than 100 with the column’s median -----
trestbps_median = df_cleaned.approxQuantile("trestbps", [0.5], 0.001)[0]
df_cleaned = df_cleaned.withColumn(
    "trestbps",
    when(col("trestbps") < 100, lit(trestbps_median)).otherwise(col("trestbps"))
)


# ----- Step 3: Replace 'oldpeak' values outside [0, 4] with the column’s median -----
oldpeak_median = df_cleaned.approxQuantile("oldpeak", [0.5], 0.001)[0]
df_cleaned = df_cleaned.withColumn(
    "oldpeak",
    when((col("oldpeak") < 0) | (col("oldpeak") > 4), lit(oldpeak_median)).otherwise(col("oldpeak"))
)


# ----- Step 4: Impute missing values for 'thaldur' and 'thalach' using median -----
thaldur_median = df_cleaned.approxQuantile("thaldur", [0.5], 0.001)[0]
thalach_median = df_cleaned.approxQuantile("thalach", [0.5], 0.001)[0]
df_cleaned = df_cleaned.na.fill({"thaldur": thaldur_median, "thalach": thalach_median})


# ----- Step 5: For 'fbs', 'prop', 'nitr', 'pro', and 'diuretic': -----
# a. Impute missing values using mode
# b. Replace any value greater than 1 with 1
for c in ['fbs', 'prop', 'nitr', 'pro', 'diuretic']:
    mode_val_row = df_cleaned.filter(col(c).isNotNull()) \
        .groupBy(c).count().orderBy("count", ascending=False).first()
    mode_val = mode_val_row[c] if mode_val_row is not None else None
    if mode_val is not None:
        df_cleaned = df_cleaned.na.fill({c: mode_val})
    df_cleaned = df_cleaned.withColumn(c, when(col(c) > 1, lit(1)).otherwise(col(c)))


# ----- Step 6: Impute missing values for 'exang' and 'slope' using mode -----
exang_mode_row = df_cleaned.filter(col("exang").isNotNull()) \
    .groupBy("exang").count().orderBy("count", ascending=False).first()
exang_mode = exang_mode_row["exang"] if exang_mode_row is not None else None

slope_mode_row = df_cleaned.filter(col("slope").isNotNull()) \
    .groupBy("slope").count().orderBy("count", ascending=False).first()
slope_mode = slope_mode_row["slope"] if slope_mode_row is not None else None

fill_dict = {}
if exang_mode is not None:
    fill_dict["exang"] = exang_mode
if slope_mode is not None:
    fill_dict["slope"] = slope_mode
if fill_dict:
    df_cleaned = df_cleaned.na.fill(fill_dict)


# Display the first 5 rows of the cleaned DataFrame
#df_cleaned.show(5)

# OUPUT IS BELOW
'''
>>> exec(open("assignment3.py").read())
+---+---+-------+--------+---+--------+-----+---+----+----+---+--------+-------+-------+-----+-------+-----+
|age|sex|painloc|painexer| cp|trestbps|smoke|fbs|prop|nitr|pro|diuretic|thaldur|thalach|exang|oldpeak|slope|
+---+---+-------+--------+---+--------+-----+---+----+----+---+--------+-------+-------+-----+-------+-----+
| 63|  1|      1|       1|  1|   145.0| null|  1|   0|   0|  0|       0|   10.5|    150|    0|    2.3|    3|
| 67|  1|      1|       1|  4|   160.0| null|  0|   1|   0|  0|       0|    9.5|    108|    1|    1.5|    2|
| 67|  1|      1|       1|  4|   120.0| null|  0|   1|   0|  0|       0|    8.5|    129|    1|    2.6|    2|
| 37|  1|      1|       1|  3|   130.0| null|  0|   1|   0|  0|       0|   13.0|    187|    0|    3.5|    3|
| 41|  0|      1|       1|  2|   130.0| null|  0|   0|   0|  0|       0|    7.0|    172|    0|    1.4|    1|
+---+---+-------+--------+---+--------+-----+---+----+----+---+--------+-------+-------+-----+-------+-----+
only showing top 5 rows

'''

# Cleaning Step 3 
'''
For Source 1, I used the "Proportion of people 15 years and over who were current daily smokers by age and sex, 2022" dataset because it allows me to impute smoking rates with a higher level of precision by considering both age and sex, which will likely yield more accurate imputation for missing values in the "smoke" column given that the heart disease dataset also has both age and sex information.

For source 2, I used the tobacco product use - cigarettes dataset and assumed that the smoking rates by different age groups for both men and women are the same.

'''
# --- Updated smoking rates for Source 1 and Source 2 ---

smoking_rates_source1 = {
    '15-17': {'Male': 1.2, 'Female': 1.8},
    '18-24': {'Male': 9.3, 'Female': 5.9},
    '25-34': {'Male': 13.4, 'Female': 8.8},
    '35-44': {'Male': 13.5, 'Female': 8.5},
    '45-54': {'Male': 15.3, 'Female': 11.6},
    '55-64': {'Male': 17.4, 'Female': 12.0},
    '65-74': {'Male': 9.9, 'Female': 7.9},
    '75+':   {'Male': 3.8, 'Female': 1.9}
}

smoking_rates_source2 = {
    '18-24': {'Male': 4.8, 'Female': 4.8},
    '25-44': {'Male': 12.5, 'Female': 12.5},
    '45-64': {'Male': 15.1, 'Female': 15.1},
    '65+':   {'Male': 8.7, 'Female': 8.7}
}

source2_smoking_rate_among_men = 13.2
source2_smoking_rate_among_women = 10.0

# --- Ensure the age column is numeric ---
df_cleaned = df_cleaned.withColumn("age", col("age").cast("double"))

# --- Define UDFs for smoking rate imputation ---

def impute_smoking_rate_source1(age, sex):
    """
    For Source 1:
      - Uses age to determine an age group.
      - Returns the corresponding smoking rate.
      - Assumes 'sex' is a string where '0' means female.
    """
    if age is None:
        return 0.0
    try:
        age = float(age)
    except Exception:
        return 0.0

    age_group = None
    if 15 <= age <= 17:
        age_group = '15-17'
    elif 18 <= age <= 24:
        age_group = '18-24'
    elif 25 <= age <= 34:
        age_group = '25-34'
    elif 35 <= age <= 44:
        age_group = '35-44'
    elif 45 <= age <= 54:
        age_group = '45-54'
    elif 55 <= age <= 64:
        age_group = '55-64'
    elif 65 <= age <= 74:
        age_group = '65-74'
    elif age >= 75:
        age_group = '75+'
    
    if age_group is None:
        return 0.0

    # If sex is '0' we treat as Female; otherwise Male.
    if sex == '0':
        return float(smoking_rates_source1[age_group]['Female'])
    else:
        return float(smoking_rates_source1[age_group]['Male'])

def impute_smoking_rate_source2(age, sex):
    """
    For Source 2:
      - Determines the age group from age.
      - For females (sex=='0'), returns the corresponding rate.
      - For males, adjusts the rate using overall male/female rates.
    """
    if age is None:
        return 0.0
    try:
        age = float(age)
    except Exception:
        return 0.0

    age_group = None
    if 18 <= age <= 24:
        age_group = '18-24'
    elif 25 <= age <= 44:
        age_group = '25-44'
    elif 45 <= age <= 64:
        age_group = '45-64'
    elif age >= 65:
        age_group = '65+'
    
    if age_group is None:
        return 0.0

    if sex == '0':  # Female
        return float(smoking_rates_source2[age_group]['Female'])
    else:  # Male: adjust the rate by the ratio of overall rates
        male_rate = smoking_rates_source2[age_group]['Male']
        return float(male_rate * (source2_smoking_rate_among_men / source2_smoking_rate_among_women))

# Register the UDFs.
udf_source1 = udf(impute_smoking_rate_source1, DoubleType())
udf_source2 = udf(impute_smoking_rate_source2, DoubleType())

# --- Apply the UDFs to create new columns ---
df_cleaned = df_cleaned.withColumn("smoke_imputed_source1", udf_source1(col("age"), col("sex")))
df_cleaned = df_cleaned.withColumn("smoke_imputed_source2", udf_source2(col("age"), col("sex")))

# --- Normalize the imputed columns ---
# For Source 1:
source1_min = df_cleaned.selectExpr("min(smoke_imputed_source1) as min_val").first()["min_val"]
source1_max = df_cleaned.selectExpr("max(smoke_imputed_source1) as max_val").first()["max_val"]

df_cleaned = df_cleaned.withColumn(
    "smoke_imputed_source1_normalized",
    (col("smoke_imputed_source1") - lit(source1_min)) / (lit(source1_max) - lit(source1_min))
)

# For Source 2:
source2_min = df_cleaned.selectExpr("min(smoke_imputed_source2) as min_val").first()["min_val"]
source2_max = df_cleaned.selectExpr("max(smoke_imputed_source2) as max_val").first()["max_val"]

df_cleaned = df_cleaned.withColumn(
    "smoke_imputed_source2_normalized",
    (col("smoke_imputed_source2") - lit(source2_min)) / (lit(source2_max) - lit(source2_min))
)

# Display the first 5 rows of the resulting DataFrame.
# df_cleaned.show(5)

# OUTPUT IS BELOW
'''
>>> exec(open("assignment3.py").read())
|67.0|  1|      1|       1|  4|   160.0| null|  0|   1|   0|  0|       0|    9.5|    108|    1|    1.5|    2|                  9.9|   11.483999999999998|              0.5689655172413793|              0.5761589403973509|      
|67.0|  1|      1|       1|  4|   120.0| null|  0|   1|   0|  0|       0|    8.5|    129|    1|    2.6|    2|                  9.9|   11.483999999999998|              0.5689655172413793|              0.5761589403973509|  |67.0|  1|      1|       1|  4|   160.0| null|  0|   1|   0|  0|       0|    9.5|    108|    1|    1.5|    2|                  9.9|   11.483999999999998|              0.5689655172413793|              0.5761589403973509|  
93|              0.5761589403973509|
|37.0|  1|      1|       1|  3|   130.0| null|  0|   1|   0|  0|       0|   13.0|    187|    0|    3.5|    3|                 13.5|   16.499999999999996|              0.7758620689655173|              0.8278145695364237|
|41.0|  0|      1|       1|  2|   130.0| null|  0|   0|   0|  0|       0|    7.0|    172|    0|    1.4|    1|                 13.5|   16.499999999999996|              0.7758620689655173|              0.8278145695364237|
+----+---+-------+--------+---+--------+-----+---+----+----+---+--------+-------+-------+-----+-------+-----+---------------------+---------------------+--------------------------------+--------------------------------+
only showing top 5 rows

'''

# ---------------- STEP 3 ----------------------------------------------------------------------------------------------

# --- Step 1: Combine features (df_cleaned) with target (from df) ---
# Add a unique row ID to both DataFrames so we can join them
df_cleaned = df_cleaned.withColumn("row_id", monotonically_increasing_id())
df_with_target = df.withColumn("row_id", monotonically_increasing_id()).select("row_id", "target")

# Join on the row_id; drop the helper column afterward
df_combined = df_cleaned.join(df_with_target, on="row_id").drop("row_id")

# Drop rows where the target is null
df_combined = df_combined.filter(col("target").isNotNull())

# --- Step 2: Stratified Split (90-10) by target ---
# Get the list of distinct target values
distinct_targets = [row["target"] for row in df_combined.select("target").distinct().collect()]

train_df = None
test_df = None

# For each target group, perform a random split and then union the results.
for t in distinct_targets:
    group_df = df_combined.filter(col("target") == t)
    splits = group_df.randomSplit([0.9, 0.1], seed=42)
    if train_df is None:
        train_df = splits[0]
        test_df = splits[1]
    else:
        train_df = train_df.union(splits[0])
        test_df = test_df.union(splits[1])

# --- Step 3: Separate features and target ---
# X_train: all columns except 'target'
X_train = train_df.drop("target")
# y_train: only the target column
y_train = train_df.select("target")
X_test = test_df.drop("target")
y_test = test_df.select("target")

# show counts for each split
print("Train set count:", train_df.count())
print("Test set count:", test_df.count())

# OUTPUT IS BELOW
'''
>>> exec(open("assignment3.py").read())
Train set count: 831
Test set count: 68

'''
df_cleaned = df_cleaned.withColumn("row_id", monotonically_increasing_id())
df_with_target = df.select("target").withColumn("row_id", monotonically_increasing_id())

# Join the cleaned features with the target column on the row_id.
data = df_cleaned.join(df_with_target, on="row_id").drop("row_id")

feature_cols = [c for c in data.columns if c != "target"]

# Cast all feature columns to double (if they are not already numeric)
for f in feature_cols:
    data = data.withColumn(f, col(f).cast("double"))

# --- Impute missing values using the median ---
imputer = Imputer(inputCols=feature_cols, outputCols=feature_cols, strategy="median")
data_imputed = imputer.fit(data).transform(data)

# Ensure the target column is not null and does not contain NaN values
data_imputed = data_imputed.filter(col("target").isNotNull() & (~isnan(col("target"))))

# Cast the target column to double (if it's not already numeric)
data_imputed = data_imputed.withColumn("target", col("target").cast("double"))

# --- Train-test split ---
# Note: randomSplit is not stratified.
train_data, test_data = data_imputed.randomSplit([0.9, 0.1], seed=42)



# ----------- TASK 4 -----------------------------------------------------------------------------------

# ----- Prepare the DataFrame for ML -----
# Assume 'data_imputed' is your DataFrame with imputed features and a numeric "target" column.
# List of feature columns (excluding the target)
feature_cols = [c for c in data_imputed.columns if c != "target"]

# Create a feature vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Optionally scale the features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# ----- Define Classifiers -----
# Logistic Regression (with increased maxIter)
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="target", maxIter=1000)

# Random Forest Classifier
rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="target")

# Linear SVC (Spark’s SVC variant; note: it does not produce probabilities)
svc = LinearSVC(featuresCol="scaledFeatures", labelCol="target", maxIter=1000)

# ----- Build Pipelines -----
# Each pipeline has the same pre-processing stages and then the classifier
pipeline_lr = Pipeline(stages=[assembler, scaler, lr])
pipeline_rf = Pipeline(stages=[assembler, scaler, rf])
pipeline_svc = Pipeline(stages=[assembler, scaler, svc])

# ----- Set Up Hyperparameter Grids -----
paramGrid_lr = (ParamGridBuilder()
                .addGrid(lr.regParam, [0.01, 0.1, 1.0])
                .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                .build())

paramGrid_rf = (ParamGridBuilder()
                .addGrid(rf.numTrees, [50, 100])
                .addGrid(rf.maxDepth, [5, 10])
                .build())

paramGrid_svc = (ParamGridBuilder()
                 .addGrid(svc.regParam, [0.01, 0.1, 1.0])
                 .build())

# ----- Set Up Evaluator -----
# Here we use areaUnderROC; you could also use accuracy or other metrics.
evaluator = BinaryClassificationEvaluator(labelCol="target", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

# ----- Set Up Cross-Validation -----
cv_lr = CrossValidator(estimator=pipeline_lr,
                       estimatorParamMaps=paramGrid_lr,
                       evaluator=evaluator,
                       numFolds=5,
                       seed=42)

cv_rf = CrossValidator(estimator=pipeline_rf,
                       estimatorParamMaps=paramGrid_rf,
                       evaluator=evaluator,
                       numFolds=5,
                       seed=42)

cv_svc = CrossValidator(estimator=pipeline_svc,
                        estimatorParamMaps=paramGrid_svc,
                        evaluator=evaluator,
                        numFolds=5,
                        seed=42)


# Fit the models with cross-validation
cvModel_lr = cv_lr.fit(train_data)
cvModel_rf = cv_rf.fit(train_data)
cvModel_svc = cv_svc.fit(train_data)

# ----- Evaluate on Test Data -----
# For predictions, transform the test set
predictions_lr = cvModel_lr.transform(test_data)
predictions_rf = cvModel_rf.transform(test_data)
predictions_svc = cvModel_svc.transform(test_data)

# Calculate ROC AUC
roc_lr = evaluator.evaluate(predictions_lr)
roc_rf = evaluator.evaluate(predictions_rf)
roc_svc = evaluator.evaluate(predictions_svc)

# For accuracy, you can use the MulticlassClassificationEvaluator
acc_evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
acc_lr = acc_evaluator.evaluate(predictions_lr)
acc_rf = acc_evaluator.evaluate(predictions_rf)
acc_svc = acc_evaluator.evaluate(predictions_svc)

print("Logistic Regression: Accuracy = {:.4f}, ROC AUC = {:.4f}".format(acc_lr, roc_lr))
print("Random Forest: Accuracy = {:.4f}, ROC AUC = {:.4f}".format(acc_rf, roc_rf))
print("Linear SVC: Accuracy = {:.4f}, ROC AUC = {:.4f}".format(acc_svc, roc_svc))

#Output

'''
Logistic Regression: Accuracy = 0.7639, ROC AUC = 0.7972
Random Forest: Accuracy = 0.7222, ROC AUC = 0.8283
Linear SVC: Accuracy = 0.7639, ROC AUC = 0.7863

'''

# ---------------- TASK 5 -----------------------------------------------------------------------------------------------------------

'''
Based on the performance metrics, while both Logistic Regression and Linear SVC achieved an accuracy of 76.39%, 
their ROC AUC scores were 0.7972 and 0.7863 respectively. In contrast, the Random Forest classifier, 
despite a slightly lower accuracy of 72.22%, achieved the highest ROC AUC of 0.8283. 

- The ROC AUC metric provides a robust measure of the model’s ability to rank positive instances higher than negatives, which is especially valuable in a binary 
classification task like predicting heart disease. 

- A higher ROC AUC suggests that the Random Forest model is better at distinguishing between the two classes, 
potentially reducing false negatives—a critical factor in medical diagnosis

- Although Logistic Regression and Linear SVC offer slightly higher accuracy, their lower ROC AUC scores indicate they may be less reliable in terms of 
overall discrimination, which is often more important than accuracy alone in imbalanced or high-stakes scenarios

Therefore, considering these relevant criteria, the Random Forest classifier is selected as my final model.

'''