import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("traffic.csv")

print(df.head())
print(df.tail())

df["holiday"] = df["holiday"].fillna("none")

print(df.isnull().sum())

df.drop_duplicates(inplace=True)

print(df.duplicated().sum())

print(df.describe())

plt.figure(figsize=(8,5))
plt.hist(df["traffic_volume"], bins=30)
plt.title("Traffic Volume Distribution")
plt.xlabel("Traffic Volume")
plt.ylabel("Frequency")
plt.show()

df["date_time"] = pd.to_datetime(df["date_time"], dayfirst=True)

df["hour"] = df["date_time"].dt.hour

hourly = df.groupby("hour")["traffic_volume"].mean().reset_index()

plt.figure(figsize=(10,5))
sns.lineplot(data=hourly, x="hour", y="traffic_volume", marker="o")
plt.title("Average Traffic Volume by Hour")
plt.xlabel("Hour")
plt.ylabel("Average Traffic Volume")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(data=df, x="weather_main", y="traffic_volume")
plt.title("Average Traffic Volume by Weather")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,5))
sns.barplot(data=df, x="holiday", y="traffic_volume")
plt.title("Traffic Volume on Holidays")
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="temp", y="traffic_volume")
plt.title("Temperature vs Traffic Volume")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="rain_1h", y="traffic_volume")
plt.title("Rain vs Traffic Volume")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="clouds_all", y="traffic_volume")
plt.title("Cloud Coverage vs Traffic Volume")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

cols = ["traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all"]

for col in cols:
    plt.figure(figsize=(8,3))
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()

df["weather_main"].value_counts().plot(kind="bar")
plt.title("Weather Frequency")
plt.show()

df["holiday"].value_counts().plot(kind="bar")
plt.title("Holiday Frequency")
plt.show()

le = LabelEncoder()

df["holiday"] = le.fit_transform(df["holiday"])
df["weather_main"] = le.fit_transform(df["weather_main"])
df["weather_description"] = le.fit_transform(df["weather_description"])

df.drop("date_time", axis=1, inplace=True)

X = df.drop("traffic_volume", axis=1)
y = df["traffic_volume"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)