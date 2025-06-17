import pandas as pd
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor

# Chargement des données prétraitées
X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv').values
y_train = pd.read_csv('data/processed_data/y_train.csv').values.ravel()

# Chargement des meilleurs hyperparamètres trouvés par GridSearch
best_params = joblib.load('models/best_params.pkl')

# Initialisation du modèle avec les meilleurs hyperparamètres
model = GradientBoostingRegressor(random_state=42, **best_params)

# Entraînement du modèle
model.fit(X_train_scaled, y_train)

# Création du dossier models si besoin
os.makedirs('models', exist_ok=True)

# Sauvegarde du modèle entraîné
joblib.dump(model, 'models/trained_model.pkl')

print("Modèle entraîné et sauvegardé dans models/trained_model.pkl")
