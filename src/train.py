import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
import os

def create_sample_data():
    """Crée des données d'exemple pour la démonstration"""
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_redundant=10, 
        random_state=42
    )
    return X, y

def train_model():
    """Entraîne le modèle et le sauvegarde"""
    print("Création des données d'entraînement...")
    X, y = create_sample_data()
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print("Entraînement du modèle...")
    model = RandomForestClassifier(n_estimators=95, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Précision du modèle: {accuracy:.4f}")
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred))
    
    # Sauvegarde du modèle
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    
    # Sauvegarde des métriques
    metrics = {
        'accuracy': accuracy,
        'model_type': 'RandomForestClassifier'
    }
    joblib.dump(metrics, 'models/metrics.pkl')
    
    print("Modèle sauvegardé dans models/model.pkl")
    return accuracy

if __name__ == "__main__":
    train_model()