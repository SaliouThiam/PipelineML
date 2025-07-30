from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Chargement du modèle au démarrage
model_path = 'models/model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de santé"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    if model is None:
        return jsonify({'error': 'Modèle non chargé'}), 500
    
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': probability[0].tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model-info', methods=['GET'])
def model_info():
    """Informations sur le modèle"""
    if model is None:
        return jsonify({'error': 'Modèle non chargé'}), 500
    
    metrics_path = 'models/metrics.pkl'
    if os.path.exists(metrics_path):
        metrics = joblib.load(metrics_path)
        return jsonify(metrics)
    else:
        return jsonify({'model_type': 'RandomForestClassifier'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)