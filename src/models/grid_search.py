import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")["silica_concentrate"]

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)

print("Meilleurs paramètres :", grid.best_params_)
joblib.dump(grid.best_params_, "models/best_params.pkl")