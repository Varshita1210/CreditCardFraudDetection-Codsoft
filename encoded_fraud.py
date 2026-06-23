import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("dataset/fraudTrain.csv")

drop_columns = [
    'Unnamed: 0',
    'first',
    'last',
    'street',
    'trans_num',
    'cc_num'
]

df = df.drop(columns=drop_columns)

for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

fraud = df[df["is_fraud"] == 1]

print(fraud.head(1).T)