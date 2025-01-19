import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train_path, y_train_path):
    """Entraîne un modèle de classification."""
    # Charger les données
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    # Initialiser et entraîner le modèle
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train.values.ravel())

    # Sauvegarder le modèle
    joblib.dump(model, "model.joblib")
    print("Modèle entraîné et sauvegardé.")

if __name__ == "__main__":
    train_model("X_train.csv", "y_train.csv")
