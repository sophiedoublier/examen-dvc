"""
normalize_data.py
-----------------
Ce script lit les fichiers X_train.csv et X_test.csv dans data/processed/,
applique une normalisation (StandardScaler) sur les features, et sauvegarde
les résultats sous X_train_scaled.csv et X_test_scaled.csv dans data/processed/.
Les chemins relatifs sont utilisés pour garantir la portabilité du script.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def main():
    input_folder = 'data/processed_data'
    output_folder = 'data/processed_data'

    # Chargement des jeux de données
    X_train = pd.read_csv(os.path.join(input_folder, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(input_folder, 'X_test.csv'))

    # Suppression de la colonne 'date' non numérique
    X_train = X_train.drop(columns=['date'])
    X_test = X_test.drop(columns=['date'])	

    # Normalisation des features (StandardScaler recommandé pour la régression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Conversion en DataFrame pour garder les noms de colonnes
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Sauvegarde des fichiers normalisés
    X_train_scaled.to_csv(os.path.join(output_folder, 'X_train_scaled.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(output_folder, 'X_test_scaled.csv'), index=False)

if __name__ == '__main__':
    main()

