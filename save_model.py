import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("dataset/fraudTrain.csv")

# Sample data
df = df.sample(n=100000, random_state=42)

# Drop columns
drop_columns = [
    'Unnamed: 0',
    'first',
    'last',
    'street',
    'trans_num',
    'cc_num'
]

df = df.drop(columns=drop_columns)

# Encode
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Features and Target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Train Model
model = DecisionTreeClassifier(
    random_state=42,
    class_weight='balanced'
)

model.fit(X, y)

# Save Model
joblib.dump(model, "models/fraud_model.pkl")

print("Model Saved Successfully!")