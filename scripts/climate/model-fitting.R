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
    X = df[["date", col]] # nolint
    
    # subset the data 
    prop = 0.8 * len(X)
    train = X[:int(prop)]
    test = X[int(prop):]

    # fit the model 
    model = IsolationForest(n_estimators=100, max_samples= 200,
                            contamination=0.005, max_features= 1,
                            random_state=42)
    model.fit(train)
    
    # predict the anomalies
    anomalies = model.predict(test)

    # plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(test["date"], test[col], label="Mean " + col)
    plt.scatter(test["date"][anomalies == -1], 
                test[col][anomalies == -1], 
                color='red', label="Anomalies")
    plt.xlabel("Days since January 1, 2013")
    plt.ylabel(col)
    plt.title(col + " Over Time with Anomalies")
    plt.legend()
    plt.show()