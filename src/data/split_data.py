"""
split_data.py
-------------
Ce script utilise le chemin relatif 'data/raw_data/raw.csv' pour garantir la portabilité et l’exécution correcte du pipeline, peu importe l’emplacement du projet.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def import_dataset(input_filepath):
    return pd.read_csv(input_filepath)

def split_data(df, test_size=0.3, random_state=42):
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        feats, target, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    os.makedirs(output_folderpath, exist_ok=True)
    X_train.to_csv(os.path.join(output_folderpath, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_folderpath, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_folderpath, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_folderpath, 'y_test.csv'), index=False)

def main():
  # Utilise un chemin relatif pour garantir la portabilité du script dans la structure du projet
    input_filepath = 'data/raw_data/raw.csv'
    output_folderpath = 'data/processed_data'
    df = import_dataset(input_filepath)
    X_train, X_test, y_train, y_test = split_data(df)
    save_dataframes(X_train, X_test, y_train, y_test, output_folderpath)

if __name__ == '__main__':
    main()
