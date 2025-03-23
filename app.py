from flask import Flask, render_template, request, jsonify, session
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from werkzeug.utils import secure_filename
import io
import base64
from datetime import datetime
from perceptron import Perceptron
from utils import load_data, plot_decision_regions, accuracy
from train import train_perceptron, train_with_visualization, split_data

app = Flask(__name__)
app.secret_key = 'perceptron_secret_key'

DATA_FOLDER = 'data'
DEFAULT_DATASET = 'sample_data.csv'

def load_scaling_params():
    with open('data/scaling_params.json', 'r') as f:
        return json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    try:
        dataset_name = request.form.get('dataset_name', DEFAULT_DATASET)
        filepath = os.path.join(DATA_FOLDER, dataset_name)

        if not os.path.exists(filepath):
            return jsonify({'error': f'Dataset {dataset_name} not found in {DATA_FOLDER} folder'}), 404
        
        model_info = train_model(filepath)

        if model_info['num_features'] == 2:
            visualization = generate_visualization(model_info['model'],
                                                   model_info['X_train'],
                                                   model_info['y_train'])
            
            model_info['visualization'] = visualization

        session['model_info'] = {
            'num_features': model_info['num_features'],
            'weights': model_info['weights'].tolist() if
isinstance(model_info['weights'], np.ndarray) else model_info['weights'],
                        'bias': float(model_info['bias'])
        }

        return jsonify({
            'accuracy': model_info['accuracy'],
            'iterations': model_info['iterations'],
            'weights': model_info['weights'].tolist() if
isinstance(model_info['weights'], np.ndarray) else model_info['weights'],
            'bias': float(model_info['bias']),
            'num_features': model_info['num_features'],
            'visualization': model_info.get('visualization')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/predict', methods=['POST'])
def predict():
    if 'model_info' not in session:
        return jsonify({'error': 'Model not trained yet'}), 400
    
    try:
        model_info = session['model_info']
        num_features = model_info['num_features']

        scaling_params = load_scaling_params()
        min_values = scaling_params['min_values']
        max_values = scaling_params['max_values']

        feature_names = list(min_values.keys())

        features = []
        for i in range(num_features):
            feature_name = f'feature_{i}'
            if feature_name in request.form:
                features.append(float(request.form[feature_name]))

        if len(features) != num_features:
            return jsonify({'error': f'Please provide {num_features} features'}), 400
        
        scaled_features = []
        for i, value in enumerate(features):
            feature_name = feature_names[i]
            min_value = min_values[feature_name]
            max_value = max_values[feature_name]

            scaled_value = (value - min_value) / (max_value - min_value)
            scaled_features.append(scaled_value)

        model = Perceptron()
        model.weights = np.array(model_info['weights'])
        model.bias = model_info['bias']

        prediction = int(model.predict(np.array(scaled_features).reshape(1, -1))[0])

        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def train_model(filepath):
    X, y = load_data(filepath)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=42)

    model, _ = train_perceptron(X_train, y_train)

    y_pred = model.predict(X_test)
    test_accuracy = accuracy(y_test, y_pred)

    return {
        'model': model,
        'accuracy': float(test_accuracy),
        'iterations': model.n_iterations,
        'weights': model.weights,
        'bias': model.bias,
        'num_features': X.shape[1],
        'X_train': X_train,
        'y_train': y_train
    }

def generate_visualization(model, X, y):
    if X.shape[1] != 2:
        return None
    
    plt.figure(figsize=(8, 6))
    plot_decision_regions(X, y, model, title='Perceptron Decision Boundary')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    encoded = base64.b64encode(image_png).decode('utf-8')
    return f'data:image/png;base64,{encoded}'

if __name__ == '__main__':
    app.run(debug=True)




    


