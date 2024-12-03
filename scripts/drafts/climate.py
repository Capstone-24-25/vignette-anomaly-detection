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
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(df)
anomalies = model.predict(df)
np.set_printoptions(threshold=np.inf)
print(anomalies)

print(anomalies[anomalies == -1].shape)

from sklearn.svm import OneClassSVM

svm = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
svm.fit(df)
anomalies = svm.predict(df)
print(anomalies)
print(anomalies[anomalies == -1].shape)