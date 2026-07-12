# main file yhi hai model training 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# Load Dataset
# ===============================

df = pd.read_csv("traffic.csv")

print("First 5 Rows")
print(df.head())

print("\nShape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

# ===============================
# Data Cleaning
# ===============================

df["holiday"] = df["holiday"].fillna("None")

df.drop_duplicates(inplace=True)

print("\nDuplicate Rows:", df.duplicated().sum())

# ===============================
# Convert Date Column
# ===============================

df["date_time"] = pd.to_datetime(df["date_time"], dayfirst=True)

df["hour"] = df["date_time"].dt.hour
df["day"] = df["date_time"].dt.day
df["month"] = df["date_time"].dt.month
df["day_of_week"] = df["date_time"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# ===============================
# EDA
# ===============================

print("\nStatistical Summary")
print(df.describe())

# Traffic Distribution
plt.figure(figsize=(8,5))
plt.hist(df["traffic_volume"], bins=30)
plt.title("Traffic Volume Distribution")
plt.xlabel("Traffic Volume")
plt.ylabel("Frequency")
plt.show()

# Hour vs Traffic
hourly = df.groupby("hour")["traffic_volume"].mean().reset_index()

plt.figure(figsize=(10,5))
sns.lineplot(data=hourly, x="hour", y="traffic_volume", marker="o")
plt.title("Average Traffic Volume by Hour")
plt.xlabel("Hour")
plt.ylabel("Average Traffic Volume")
plt.grid(True)
plt.show()

# Weather vs Traffic
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="weather_main", y="traffic_volume")
plt.xticks(rotation=45)
plt.title("Average Traffic Volume by Weather")
plt.show()

# Correlation
plt.figure(figsize=(10,7))
sns.heatmap(df.select_dtypes(include="number").corr(),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# ===============================
# Label Encoding
# ===============================

le = LabelEncoder()

df["holiday"] = le.fit_transform(df["holiday"])
df["weather_main"] = le.fit_transform(df["weather_main"])
df["weather_description"] = le.fit_transform(df["weather_description"])

# ===============================
# Drop Date Column
# ===============================

df.drop("date_time", axis=1, inplace=True)

# ===============================
# Features & Target
# ===============================

X = df.drop("traffic_volume", axis=1)
y = df["traffic_volume"]

# ===============================
# Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# ===============================
# Random Forest Model
# ===============================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# Prediction
# ===============================

y_pred = model.predict(X_test)

# ===============================
# Evaluation
# ===============================

print("\n========== MODEL PERFORMANCE ==========")

print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score :", r2_score(y_test, y_pred))

# ===============================
# Actual vs Predicted
# ===============================

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted")
print(comparison.head(20))

# ===============================
# Prediction Plot
# ===============================

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Traffic")
plt.ylabel("Predicted Traffic")
plt.title("Actual vs Predicted Traffic")
plt.show()

# ===============================
# Feature Importance
# ===============================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(by="Importance", ascending=False)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(10,5))
sns.barplot(data=importance, x="Importance", y="Feature")
plt.title("Feature Importance")
plt.show()

# ===============================
# Save Model
# ===============================

joblib.dump(model, "traffic_prediction_model.pkl")

print("\nModel Saved Successfully!")
# now this model comes to end by Aditya Maurya 
