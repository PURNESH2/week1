# Classification of Fire Types in India Using MODIS Satellite Data
# Refactored Python Script

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  accuracy_score,classification_report,ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

################################################################################

df1 = pd.read_csv('modis_2021_India.csv')
df2 = pd.read_csv('modis_2022_India.csv')
df3 = pd.read_csv('modis_2023_India.csv')

################################################################################

df1.head() # print first 5 rows - df1.tail()

################################################################################

df2.head()

################################################################################

df3.head()

################################################################################

df = pd.concat([df1, df2, df3], ignore_index=True)
df.head()

################################################################################

df.shape # rows and cols

################################################################################

df.info() # dt, memc

################################################################################

# Any missing values?
df.isnull().sum()

################################################################################

df.duplicated().sum()

################################################################################

# List out column names to check
df.columns

################################################################################

df.describe().T # statistics of dataset - numbers!

################################################################################

# Check Unique values of target variable
df.type.value_counts()

################################################################################

# Check unique and n unique for all categorical features
for col in df.columns:
  if df[col].dtype == 'object':
    print(f"Column: {col}")
    print(f"Unique values: {df[col].unique()}")
    print(f"Number of unique values: {df[col].nunique()}")
    print("-" * 50)

################################################################################

# Count plot for 'type'
plt.figure(figsize=(8, 6))
sns.countplot(x='type', data=df)
plt.title('Distribution of Fire Types')
plt.xlabel('Fire Type')
plt.ylabel('Count')
plt.show()

################################################################################

# Histogram of 'confidence'
plt.figure(figsize=(8, 6))
sns.histplot(df['confidence'], bins=20, kde=True)
plt.title('Distribution of Confidence')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.show()

################################################################################
