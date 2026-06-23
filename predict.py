import joblib
import numpy as np

# Load model
model = joblib.load("models/fraud_model.pkl")

# Sample transaction
sample = np.array([[

514,      # trans_date_trans_time
8,        # merchant
4,        # category
4.97,     # amt
1,        # gender
241,      # city
39,       # state
29209,    # zip
36.0788,  # lat
-81.1781, # long
3495,     # city_pop
275,      # job
514,      # dob
1325376018, # unix_time
36.0112,  # merch_lat
-82.0483  # merch_long

]])

prediction = model.predict(sample)

if prediction[0] == 0:
    print("Legitimate Transaction")
else:
    print("Fraudulent Transaction")