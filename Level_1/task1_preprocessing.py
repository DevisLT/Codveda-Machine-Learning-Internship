import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv("datasets/churn_train.csv")
test_df  = pd.read_csv("datasets/churn_test.csv")
df = pd.concat([train_df, test_df], ignore_index=True)

print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn types:\n{df.dtypes}")

missing = df.isnull().sum()
print(f"\nMissing values:\n{missing[missing > 0] if missing.sum() > 0 else 'None found'}")

np.random.seed(42)
for col in ['Account length', 'Total day minutes', 'Total intl charge']:
    idx = np.random.choice(df.index, size=50, replace=False)
    df.loc[idx, col] = np.nan

df['Account length']    = df['Account length'].fillna(df['Account length'].median())
df['Total day minutes'] = df['Total day minutes'].fillna(df['Total day minutes'].mean())
df['Total intl charge'] = df['Total intl charge'].fillna(df['Total intl charge'].median())

le = LabelEncoder()
df['International plan'] = le.fit_transform(df['International plan'].astype(str))
df['Voice mail plan']    = le.fit_transform(df['Voice mail plan'].astype(str))
df['Churn']              = df['Churn'].astype(int)
df = pd.get_dummies(df, columns=['State'], drop_first=True)

print(f"\nAfter encoding: {df.shape}")

numerical_cols = [
    'Account length', 'Number vmail messages',
    'Total day minutes', 'Total day calls', 'Total day charge',
    'Total eve minutes', 'Total eve calls', 'Total eve charge',
    'Total night minutes', 'Total night calls', 'Total night charge',
    'Total intl minutes', 'Total intl calls', 'Total intl charge',
    'Customer service calls'
]

X = df.drop(columns=['Churn'])
y = df['Churn']

scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set : {X_test.shape[0]} samples")
print(f"Churn rate  : {y.mean():.2%}")

X_train.to_csv("datasets/X_train.csv", index=False)
X_test.to_csv("datasets/X_test.csv", index=False)
y_train.to_csv("datasets/y_train.csv", index=False)
y_test.to_csv("datasets/y_test.csv", index=False)

print("\nPreprocessing complete. Files saved.")
