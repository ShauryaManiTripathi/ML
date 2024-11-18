import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model

# Load the dataset (for preprocessing pipeline)
data = pd.read_csv('insurance.csv')

# Preprocessing pipeline (must match training process)
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
data['sex'] = le_sex.fit_transform(data['sex'])  # Male=1, Female=0
data['smoker'] = le_smoker.fit_transform(data['smoker'])  # Yes=1, No=0
data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Standardize features (use the same scaler settings)
X = data.drop(['charges'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a function for preprocessing test samples
def preprocess_test_sample(sample, scaler, le_sex, le_smoker, feature_columns):
    # Convert categorical values to encoded form
    sample['sex'] = le_sex.transform([sample['sex']])[0]
    sample['smoker'] = le_smoker.transform([sample['smoker']])[0]
    
    # Create dummy variables for 'region' and ensure alignment with training features
    sample = pd.DataFrame([sample])  # Convert to DataFrame
    sample = pd.get_dummies(sample, columns=['region'], drop_first=True)
    for col in feature_columns:
        if col not in sample:
            sample[col] = 0  # Add missing columns with default value 0
    
    # Standardize numerical values
    sample_scaled = scaler.transform(sample[feature_columns])
    return sample_scaled

# Load the best model (update the file path if needed)
best_model_file = 'model_fold_3.keras'  # Replace with the actual best model file
model = load_model(best_model_file)

# Test sample (ensure the keys match the original dataset columns)
test_sample = {
    'age': 19,
    'sex': 'female',  # 'female' or 'male'
    'bmi': 27.9,
    'children': 0,
    'smoker': 'yes',  # 'yes' or 'no'
    'region': 'southwest'  # One of: 'southwest', 'southeast', 'northwest', 'northeast'
}

# Preprocess the test sample
feature_columns = X.columns.tolist()  # Columns used during training
test_sample_scaled = preprocess_test_sample(test_sample, scaler, le_sex, le_smoker, feature_columns)

# Make a prediction
predicted_charges = model.predict(test_sample_scaled)
print(f"Predicted Insurance Charges: ${predicted_charges[0][0]:.2f}")
