from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model_file = 'model_fold_3.keras'  # Replace with the best model file
model = load_model(model_file)

# Load the dataset (for preprocessing pipeline)
data = pd.read_csv('insurance.csv')

# Preprocessing pipeline (same as training)
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
data['sex'] = le_sex.fit_transform(data['sex'])  # Male=1, Female=0
data['smoker'] = le_smoker.fit_transform(data['smoker'])  # Yes=1, No=0
data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Standardize features
X = data.drop(['charges'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define feature columns
feature_columns = X.columns.tolist()

# Preprocessing function for test samples
def preprocess_test_sample(sample):
    sample['sex'] = le_sex.transform([sample['sex']])[0]
    sample['smoker'] = le_smoker.transform([sample['smoker']])[0]
    
    # Convert to DataFrame and align columns
    sample = pd.DataFrame([sample])
    sample = pd.get_dummies(sample, columns=['region'], drop_first=True)
    for col in feature_columns:
        if col not in sample:
            sample[col] = 0  # Add missing columns
    
    # Standardize
    sample_scaled = scaler.transform(sample[feature_columns])
    return sample_scaled

# Define a threshold for "high" vs. "low" premium
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

        # Preprocess the input
        preprocessed_input = preprocess_test_sample(user_input)

        # Predict using the model
        predicted_charges = model.predict(preprocessed_input)[0][0]

        # Determine premium status
        premium_status = "High" if predicted_charges > PREMIUM_THRESHOLD else "Low"
        print(predicted_charges)
        # Pass results to the result page
        return render_template(
            'result.html',
            charges=f"${predicted_charges:.2f}",
            premium_status=premium_status
        )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
