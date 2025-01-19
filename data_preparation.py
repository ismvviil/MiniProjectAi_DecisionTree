# data_preparation.py
# Description : Script pour préparer les données du projet
# Auteur : [Votre Nom]
# Date : [Date du jour]

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    try:
        dataset = pd.read_csv(file_path)
        print("Données chargées avec succès.")
        return dataset
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return None

def preprocess_data(dataset):
    """Prépare et nettoie les données."""
    # Gérer les valeurs manquantes
    dataset['Satisfaction Level'].fillna('Neutral', inplace=True)

    # Encodage des variables catégoriques
    label_encoders = {}
    for column in ['Gender', 'City', 'Membership Type', 'Satisfaction Level']:
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])
        label_encoders[column] = le

    return dataset, label_encoders

def split_data(dataset, features, target):

    X = dataset[features]
    y = dataset[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Données divisées en ensembles d'entraînement et de test.")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Chemin du fichier de données
    file_path = "dataset.csv"

    # Charger les données
    dataset = load_data(file_path)

    if dataset is not None:
        # Prétraiter les données
        dataset, label_encoders = preprocess_data(dataset)

        # Sélectionner les caractéristiques et la cible
        features = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Discount Applied', 'Days Since Last Purchase']
        target = 'Satisfaction Level'

        # Diviser les données
        X_train, X_test, y_train, y_test = split_data(dataset, features, target)

        # Enregistrer les données préparées
        X_train.to_csv("X_train.csv", index=False)
        X_test.to_csv("X_test.csv", index=False)
        y_train.to_csv("y_train.csv", index=False)
        y_test.to_csv("y_test.csv", index=False)
        print("Données préparées et enregistrées.")
