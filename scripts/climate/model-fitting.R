import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv("../../data/raw/DailyDelhiClimateTrain.csv")

# Convert from datetime to floating point
df["date"] = pd.to_datetime(df["date"])

# Turn date into floating point
df["date"] = df["date"].astype(np.int64) / (10**18)

# Normalize the date column to be between 1 and 1462
range = df["date"].max() - df["date"].min()
df["date"] = df["date"] - df["date"].min()
df["date"] = df["date"] / range
df["date"] = df["date"] * 1461 + 1


for col in df.columns[1:]:
    X = df[["date", col]]
    y = df["meantemp"]
    X_train, X_test, _, _ = train_test_split(X, y, random_state=42)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)
    anomalies = model.predict(X_test)
    # Sort the test results by date
    X_test_sorted = X_test.sort_values(by="date")
    anomalies_sorted = anomalies[np.argsort(X_test["date"])]

    # plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(X_test_sorted["date"], X_test_sorted[col], label="Mean " + col)
    plt.scatter(X_test_sorted["date"][anomalies_sorted == -1], 
                X_test_sorted[col][anomalies_sorted == -1], 
                color='red', label="Anomalies")
    plt.xlabel("Days since January 1, 2013")
    plt.ylabel(col)
    plt.title(col + " Over Time with Anomalies")
    plt.legend()
    plt.show()