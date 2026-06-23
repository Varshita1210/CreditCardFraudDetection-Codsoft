import pandas as pd

df = pd.read_csv("dataset/fraudTrain.csv")

fraud = df[df["is_fraud"] == 1]

print(fraud.head(1).T)