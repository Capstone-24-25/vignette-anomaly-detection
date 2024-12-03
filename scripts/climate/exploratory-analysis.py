import kagglehub
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

path = kagglehub.dataset_download("sumanthvrao/daily-climate-time-series-data")

# Load the dataset
df = pd.read_csv(path + "/DailyDelhiClimateTrain.csv")

# Check for missing values
print(df.isnull().sum())

# Convert the date column to a datetime object
df["date"] = pd.to_datetime(df["date"])
df["date"] = df["date"].astype(np.int64) / (10**18)

# Normalize the date column to be between 1 and 1462
range = df["date"].max() - df["date"].min()
df["date"] = df["date"] - df["date"].min()
df["date"] = df["date"] / range
df["date"] = df["date"] * 1461 + 1


# Plot the data
for col in df.columns[1:]:
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df[col], label = col)
    plt.xlabel("Days since January 1st 2013")
    plt.ylabel(col)
    plt.title(f"{col} Over Time")
    plt.legend()
    plt.show()
