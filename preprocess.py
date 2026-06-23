import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("dataset/fraudTrain.csv")

# Remove unnecessary columns
drop_columns = [
    'Unnamed: 0',
    'first',
    'last',
    'street',
    'trans_num',
    'cc_num'
]

df = df.drop(columns=drop_columns)

# Convert categorical columns into numbers
label_encoder = LabelEncoder()

categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)