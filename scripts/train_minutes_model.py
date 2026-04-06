import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pickle

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv("data/features.csv")

# -------------------------
# Select Features + Target
# -------------------------
features = [
    'last5_minutes',
    'last5_points',
    'rest_days',
    'home_game'
]

target = 'minutes'

# Drop missing values
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Train Model
# -------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------
# Evaluate Model
# -------------------------
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print(f"MAE: {mae:.2f} minutes")

# -------------------------
# Save Model
# -------------------------
import os
os.makedirs("models", exist_ok=True)

with open("models/minutes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to models/minutes_model.pkl")