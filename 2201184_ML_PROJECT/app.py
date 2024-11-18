from flask import Flask, render_template, request
import pandas as pd
import pickle
import logging
import os
from datetime import datetime  # Added the missing import

# Initialize Flask app
app = Flask(__name__)

# Set up logging for the Flask app
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/flask_app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Load the trained model and scaler
with open('best_linear_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']

# Load the encoders
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
    le_sex = encoders['sex']
    le_smoker = encoders['smoker']

# Load feature columns
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

def preprocess_test_sample(sample):
    """Preprocess a single test sample."""
    # Encode categorical variables
    sample['sex'] = le_sex.transform([sample['sex']])[0]
    sample['smoker'] = le_smoker.transform([sample['smoker']])[0]
    
    # Convert to DataFrame and handle one-hot encoding
    sample = pd.DataFrame([sample])
    sample = pd.get_dummies(sample, columns=['region'], drop_first=True)
    
    # Ensure all columns are present
    for col in feature_columns:
        if col not in sample:
            sample[col] = 0
    
    # Reorder columns to match training data
    sample = sample[feature_columns]
    
    # Scale features
    sample_scaled = scaler.transform(sample)
    return sample_scaled

# Define threshold for "high" vs "low" premium
PREMIUM_THRESHOLD = 10000

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Extract input data from form
        user_input = {
            'age': int(request.form['age']),
            'sex': request.form['sex'],
            'bmi': float(request.form['bmi']),
            'children': int(request.form['children']),
            'smoker': request.form['smoker'],
            'region': request.form['region']
        }
        
        logging.info(f"Received user input: {user_input}")

        # Preprocess the input
        preprocessed_input = preprocess_test_sample(user_input)

        # Predict using the model
        predicted_charges = model.predict(preprocessed_input)[0]
        
        # Determine premium status
        premium_status = "High" if predicted_charges > PREMIUM_THRESHOLD else "Low"
        
        logging.info(f"Predicted charges: ${predicted_charges:.2f}")
        
        # Pass results to the result page
        return render_template(
            'result.html',
            charges=f"${predicted_charges:.2f}",
            premium_status=premium_status
        )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)