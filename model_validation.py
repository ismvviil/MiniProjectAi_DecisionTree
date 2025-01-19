# model_validation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_prepare_data(file_path):
    """Charge et prépare les données avec validation."""
    # Charger les données
    dataset = pd.read_csv(file_path)
    
    # Séparer les caractéristiques numériques et catégorielles
    numeric_features = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 
                       'Discount Applied', 'Days Since Last Purchase']
    categorical_features = ['Gender', 'City', 'Membership Type']
    
    # Créer un dictionnaire pour stocker les encodeurs
    label_encoders = {}
    
    # Encoder les variables catégorielles
    for feature in categorical_features:
        if feature in dataset.columns:
            label_encoders[feature] = LabelEncoder()
            dataset[feature] = label_encoders[feature].fit_transform(dataset[feature])
    
    # Encoder la variable cible (Satisfaction Level)
    label_encoders['Satisfaction Level'] = LabelEncoder()
    dataset['Satisfaction Level'] = label_encoders['Satisfaction Level'].fit_transform(dataset['Satisfaction Level'])
    
    # Sélectionner toutes les caractéristiques
    X = dataset[numeric_features + categorical_features]
    y = dataset['Satisfaction Level']
    
    return X, y, dataset, numeric_features, categorical_features, label_encoders

def perform_data_quality_checks(dataset, X_train, numeric_features):
    """Effectue des vérifications de qualité des données."""
    print("\n=== Vérification de la qualité des données ===")
    
    # Vérifier les distributions des caractéristiques numériques
    print("\nRésumé statistique des caractéristiques numériques:")
    print(X_train[numeric_features].describe())
    
    try:
        # Vérifier les corrélations (uniquement pour les variables numériques)
        numeric_data = dataset[numeric_features + ['Satisfaction Level']].copy()
        numeric_correlation = numeric_data.corr()
        print("\nCorrélations avec la satisfaction (variables numériques):")
        print(numeric_correlation['Satisfaction Level'].sort_values(ascending=False))
        
        # Visualiser la matrice de corrélation des variables numériques
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_correlation, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Matrice de Corrélation (Variables Numériques)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"\nErreur lors du calcul des corrélations: {e}")
    
    # Vérifier les valeurs uniques
    print("\nNombre de valeurs uniques par colonne:")
    for column in X_train.columns:
        print(f"{column}: {X_train[column].nunique()} valeurs uniques")

def validate_model(X, y):
    """Effectue une validation croisée et entraîne un modèle régularisé."""
    # Diviser les données avec stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Créer un modèle avec paramètres anti-surapprentissage
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )
    
    # Effectuer la validation croisée
    print("\n=== Résultats de la validation croisée ===")
    scores = cross_val_score(model, X, y, cv=5)
    print("Scores de validation croisée:", scores)
    print(f"Score moyen: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    # Entraîner le modèle final
    model.fit(X_train, y_train)
    
    # Vérifier l'équilibre des classes
    print("\n=== Distribution des classes ===")
    print(y_train.value_counts(normalize=True))
    
    # Analyser les probabilités de prédiction
    predictions = model.predict(X_test)
    
    # Afficher le rapport de classification
    print("\n=== Rapport de classification ===")
    print(classification_report(y_test, predictions))
    
    return model, X_train, X_test, y_train, y_test

def visualize_feature_importance(model, X):
    """Visualise l'importance des caractéristiques."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Importance des caractéristiques")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Chemin du fichier de données
    file_path = "dataset.csv"
    
    try:
        # Charger et préparer les données
        X, y, dataset, numeric_features, categorical_features, label_encoders = load_and_prepare_data(file_path)
        
        # Effectuer les vérifications de qualité
        perform_data_quality_checks(dataset, X, numeric_features)
        
        # Valider et entraîner le modèle
        model, X_train, X_test, y_train, y_test = validate_model(X, y)
        
        # Visualiser l'importance des caractéristiques
        visualize_feature_importance(model, X)
        
        # Sauvegarder le modèle validé et les encodeurs
        joblib.dump(model, "validated_model.joblib")
        joblib.dump(label_encoders, "label_encoders.joblib")
        print("\nModèle et encodeurs sauvegardés dans 'validated_model.joblib' et 'label_encoders.joblib'")
        
    except Exception as e:
        print(f"Une erreur s'est produite : {str(e)}")