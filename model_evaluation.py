# model_evaluation.py
# Description : Script pour évaluer le modèle
# Auteur : [Votre Nom]
# Date : [Date du jour]

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt
import joblib

def evaluate_model(X_test_path, y_test_path, model_path):
    """Évalue un modèle entraîné."""
    # Charger les données
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    # Charger le modèle
    model = joblib.load(model_path)

    # Faire des prédictions
    y_pred = model.predict(X_test)

    # Afficher le rapport de classification
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))

    # Générer et afficher la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print("Matrice de Confusion :")
    print(cm)

    # Afficher la matrice de confusion sous forme graphique
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title("Matrice de Confusion")
    plt.show()

    # Visualiser l'arbre de décision si applicable
    if hasattr(model, "estimators_") or hasattr(model, "tree_"):
        print("Arbre de décision :")
        if hasattr(model, "estimators_"):
            # Random Forest: visualiser le premier arbre
            estimator = model.estimators_[0]
            plt.figure(figsize=(30, 20))  # Taille augmentée pour une meilleure visualisation
            plot_tree(
                estimator, 
                filled=True, 
                feature_names=X_test.columns, 
                class_names=[str(c) for c in model.classes_], 
                rounded=True, 
                fontsize=12
            )
            plt.title("Visualisation améliorée de l'arbre de décision (premier arbre)", fontsize=16)
            plt.show()

            # Ajout du graphique de l'importance des caractéristiques
            feature_importances = model.feature_importances_
            feature_names = X_test.columns
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(10, 6))
            plt.bar(importance_df['Feature'], importance_df['Importance'])
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('Importance des caractéristiques (Random Forest)')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()

        elif hasattr(model, "tree_"):
            # Single Decision Tree
            plt.figure(figsize=(30, 20))
            plot_tree(
                model, 
                filled=True, 
                feature_names=X_test.columns, 
                class_names=[str(c) for c in model.classes_], 
                rounded=True, 
                fontsize=12
            )
            plt.title("Visualisation améliorée de l'arbre de décision", fontsize=16)
            plt.show()

if __name__ == "__main__":
    evaluate_model("X_test.csv", "y_test.csv", "model.joblib")