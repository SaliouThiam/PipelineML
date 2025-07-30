import unittest
import sys
import os
import joblib
import numpy as np

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import train_model, create_sample_data

class TestModel(unittest.TestCase):
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.model_path = 'models/model.pkl'
        self.metrics_path = 'models/metrics.pkl'
    
    def test_model_training(self):
        """Test d'entraînement du modèle"""
        accuracy = train_model()
        self.assertGreater(accuracy, 0.7)  # Précision minimale attendue
        self.assertTrue(os.path.exists(self.model_path))
    
    def test_model_prediction(self):
        """Test de prédiction du modèle"""
        if os.path.exists(self.model_path):
            model = joblib.load(self.model_path)
            X, _ = create_sample_data()
            predictions = model.predict(X[:5])
            self.assertEqual(len(predictions), 5)
            self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_model_metrics(self):
        """Test de sauvegarde des métriques"""
        if os.path.exists(self.metrics_path):
            metrics = joblib.load(self.metrics_path)
            self.assertIn('accuracy', metrics)
            self.assertIn('model_type', metrics)

if __name__ == '__main__':
    unittest.main()