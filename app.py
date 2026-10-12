import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load files on startup
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scalar.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('mappings.pkl', 'rb') as f:
    maps = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Extract inputs with defaults
        year = int(data.get('year', 2026))
        university_id = str(data.get('university_id', ''))
        programme = str(data.get('programme', ''))
        category = str(data.get('category', ''))
        round_val = str(data.get('round', ''))
        
        # Build interaction keys
        univ_subject = f"{university_id}_{programme}"
        univ_subject_cat = f"{univ_subject}_{category}"
        univ_subject_cat_round = f"{univ_subject_cat}_{round_val}"
        
        # Map using loaded target encoding dictionaries
        g_mean = maps['global_mean']
        univ_subject_encoded = maps['us_means'].get(univ_subject, g_mean)
        univ_subject_cat_encoded = maps['usc_means'].get(univ_subject_cat, g_mean)
        univ_subject_cat_round_encoded = maps['uscr_means'].get(univ_subject_cat_round, g_mean)
        
        # Format input feature vector
        features_df = pd.DataFrame([[
            year,
            univ_subject_encoded,
            univ_subject_cat_encoded,
            univ_subject_cat_round_encoded
        ]], columns=['year', 'univ_subject_encoded', 'univ_subject_cat_encoded', 'univ_subject_cat_round_encoded'])
        
        # Apply scale transformation (if needed/designed)
        scaled_features = scaler.transform(features_df)
        
        # Generate prediction
        prediction = model.predict(scaled_features)
        
        return jsonify({
            'status': 'success',
            'predicted_cutoff': float(prediction[0])
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
