import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")["silica_concentrate"]

best_params = joblib.load("models/best_params.pkl")

model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "models/random_forest_regressor.pkl")