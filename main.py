import pandas as pd

train_df = pd.read_csv("dataset/fraudTrain.csv")

print("Shape:", train_df.shape)

print("\nFirst 5 Rows:")
print(train_df.head())

print("\nDataset Information:")
print(train_df.info())

print("\nMissing Values:")
print(train_df.isnull().sum())

print("\nFraud Distribution:")
print(train_df["is_fraud"].value_counts())