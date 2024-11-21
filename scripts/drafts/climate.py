import kagglehub
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
# Download latest version
path = kagglehub.dataset_download("sumanthvrao/daily-climate-time-series-data")

# Load the dataset
df = pd.read_csv(path + "/DailyDelhiClimateTrain.csv")
df["date"] = pd.to_datetime(df["date"])

# Turn date into floating point
df["date"] = df["date"].astype(np.int64) / (10**9)

print(df.head())
# Start anomaly detection
model = IsolationForest()
model.fit(df)
anomalies = model.predict(df)
print(anomalies)
