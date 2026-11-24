import json
import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")["silica_concentrate"]

model = joblib.load("models/random_forest_regressor.pkl")
y_pred = model.predict(X_test)

predictions = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
predictions.to_csv("data/predictions.csv", index=False)

scores = {
    "mse": mean_squared_error(y_test, y_pred),
    "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
    "mae": mean_absolute_error(y_test, y_pred),
    "r2": r2_score(y_test, y_pred),
}

with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=4)