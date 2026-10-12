import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the single pipeline file
with open('model.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Prepare raw input matching features used during training
        input_data = pd.DataFrame([{
            'year': int(data.get('year', 2026)),
            'university_id': str(data.get('university_id', 'missing')),
            'programme': str(data.get('programme', 'missing')),
            'category': str(data.get('category', 'missing')),
            'round': str(data.get('round', 'missing'))
        }])
        
        # Predict directly using the pipeline (TargetEncoder is run automatically!)
        prediction = model_pipeline.predict(input_data)
        
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
