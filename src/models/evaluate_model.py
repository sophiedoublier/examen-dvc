import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Chargement des données de test
X_test_scaled = pd.read_csv('data/processed_data/X_test_scaled.csv').values
y_test = pd.read_csv('data/processed_data/y_test.csv').values.ravel()

# Chargement du modèle entraîné
model = joblib.load('models/trained_model.pkl')

# Prédictions
y_pred = model.predict(X_test_scaled)

# Calcul des métriques
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Création du dossier metrics si besoin
os.makedirs('metrics', exist_ok=True)

# Sauvegarde des scores dans un fichier JSON
scores = {
    'mse': mse,
    'rmse': rmse,
    'mae': mae,
    'r2': r2
}
with open('metrics/scores.json', 'w') as f:
    json.dump(scores, f, indent=4)

# Création du dossier data/predictions si besoin
os.makedirs('data/predictions', exist_ok=True)

# Sauvegarde des prédictions dans un fichier CSV
df_pred = pd.DataFrame({
    'y_test': y_test,
    'y_pred': y_pred
})
df_pred.to_csv('data/predictions/test_predictions.csv', index=False)

print("Évaluation terminée. Scores sauvegardés dans metrics/scores.json et prédictions dans data/predictions/test_predictions.csv.")
