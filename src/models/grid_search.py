import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import os

# Chargement des données normalisées
X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv').values
y_train = pd.read_csv('data/processed_data/y_train.csv').values

# Si y_train est un tableau 2D de forme (n,1), le transformer en 1D
y_train = y_train.ravel()

# Définition du modèle et de la grille d'hyperparamètres
model = GradientBoostingRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

# Configuration du GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

# Exécution de la recherche
grid_search.fit(X_train_scaled, y_train)

# Affichage des meilleurs paramètres
print("Meilleurs hyperparamètres :", grid_search.best_params_)

# Sauvegarde des meilleurs paramètres dans models/best_params.pkl
os.makedirs('models', exist_ok=True)
joblib.dump(grid_search.best_params_, 'models/best_params.pkl')

