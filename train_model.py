import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("dataset/fraudTrain.csv")

# Take 100000 random samples
df = df.sample(n=100000, random_state=42)

# Drop unnecessary columns
drop_columns = [
    'Unnamed: 0',
    'first',
    'last',
    'street',
    'trans_num',
    'cc_num'
]

df = df.drop(columns=drop_columns)

# Encode categorical columns
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Features and Target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Train-Test Split with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

# Logistic Regression with balanced class weights
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))