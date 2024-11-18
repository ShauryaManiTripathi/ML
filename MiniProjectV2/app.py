import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)

class InsurancePremiumPredictor:
    def __init__(self):
        self.le_sex = LabelEncoder()
        self.le_smoker = LabelEncoder()
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = float('inf')
        self.feature_columns = None
        self.feature_importance = None
        
    def preprocess_data(self, data):
        df = data.copy()
        df['sex'] = self.le_sex.fit_transform(df['sex'])
        df['smoker'] = self.le_smoker.fit_transform(df['smoker'])
        df = pd.get_dummies(df, columns=['region'], drop_first=True)
        self.feature_columns = df.drop(['charges'], axis=1).columns
        X = df.drop(['charges'], axis=1)
        X_scaled = self.scaler.fit_transform(X)
        y = df['charges']
        return X_scaled, y
    
    def train_models(self, X, y, cv):
        models = {
            'XGBoost': (xgb.XGBRegressor(), {
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5]
            }),
            'LightGBM': (lgb.LGBMRegressor(), {
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 62, 93],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }),
            'RandomForest': (RandomForestRegressor(), {
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [3, 4, 5, 6, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt']
            })
        }
        
        best_model = None
        best_score = float('inf')
        best_name = None
        
        for name, (model, params) in models.items():
            random_search = RandomizedSearchCV(
                model, param_distributions=params,
                n_iter=20, cv=cv, scoring='neg_mean_squared_error',
                random_state=42, n_jobs=-1
            )
            random_search.fit(X, y)
            score = -random_search.best_score_
            print(f"{name} MSE: {score:.2f}")
            
            if score < best_score:
                best_score = score
                best_model = random_search.best_estimator_
                best_name = name
        
        return best_model, best_score, best_name
    
    def calculate_feature_importance(self):
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            self.feature_importance = dict(zip(self.feature_columns, importance))
        return self.feature_importance
    
    def train(self, data_path, n_splits=5):
        data = pd.read_csv(data_path)
        X, y = self.preprocess_data(data)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.best_model, self.best_score, self.best_model_name = self.train_models(X, y, kf)
        self.calculate_feature_importance()
        return self.best_model_name, self.best_score
    
    def predict(self, sample):
        if self.best_model is None:
            raise ValueError("Model hasn't been trained yet!")
        
        sample_df = pd.DataFrame([sample])
        sample_df['sex'] = self.le_sex.transform(sample_df['sex'])
        sample_df['smoker'] = self.le_smoker.transform(sample_df['smoker'])
        sample_df = pd.get_dummies(sample_df, columns=['region'], drop_first=True)
        
        for col in self.feature_columns:
            if col not in sample_df.columns:
                sample_df[col] = 0
        
        sample_scaled = self.scaler.transform(sample_df[self.feature_columns])
        prediction = self.best_model.predict(sample_scaled)
        return prediction[0]
    
    def save_model(self, path):
        if self.best_model is None:
            raise ValueError("No model to save!")
        
        model_data = {
            'model': self.best_model,
            'le_sex': self.le_sex,
            'le_smoker': self.le_smoker,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_name': self.best_model_name,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load_model(cls, path):
        model_data = joblib.load(path)
        predictor = cls()
        predictor.best_model = model_data['model']
        predictor.le_sex = model_data['le_sex']
        predictor.le_smoker = model_data['le_smoker']
        predictor.scaler = model_data['scaler']
        predictor.feature_columns = model_data['feature_columns']
        predictor.best_model_name = model_data['model_name']
        predictor.feature_importance = model_data['feature_importance']
        return predictor

# Initialize predictor
predictor = None
PREMIUM_THRESHOLD = 10000
MODEL_PATH = 'insurance_model.joblib'

def create_feature_importance_plot():
    if predictor.feature_importance is None:
        return None
    
    importance_df = pd.DataFrame({
        'Feature': list(predictor.feature_importance.keys()),
        'Importance': list(predictor.feature_importance.values())
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                 title='Feature Importance')
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = {
            'age': int(request.form['age']),
            'sex': request.form['sex'],
            'bmi': float(request.form['bmi']),
            'children': int(request.form['children']),
            'smoker': request.form['smoker'],
            'region': request.form['region']
        }
        
        predicted_charges = predictor.predict(user_input)
        premium_status = "High" if predicted_charges > PREMIUM_THRESHOLD else "Low"
        
        feature_importance_plot = create_feature_importance_plot()
        
        return render_template(
            'result.html',
            charges=f"${predicted_charges:.2f}",
            premium_status=premium_status,
            model_name=predictor.best_model_name,
            feature_importance_plot=feature_importance_plot
        )
    
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        prediction = predictor.predict(data)
        return jsonify({
            'prediction': prediction,
            'premium_status': "High" if prediction > PREMIUM_THRESHOLD else "Low",
            'model_used': predictor.best_model_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        data_path = request.json.get('data_path', 'insurance.csv')
        n_splits = request.json.get('n_splits', 5)
        
        best_model_name, best_score = predictor.train(data_path, n_splits)
        predictor.save_model(MODEL_PATH)
        
        return jsonify({
            'status': 'success',
            'best_model': best_model_name,
            'mse_score': best_score
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/feature_importance', methods=['GET'])
def api_feature_importance():
    if predictor.feature_importance is None:
        return jsonify({'error': 'No feature importance available'}), 400
    return jsonify(predictor.feature_importance)

def initialize_app():
    global predictor
    try:
        predictor = InsurancePremiumPredictor.load_model(MODEL_PATH)
        print("Loaded existing model")
    except:
        print("Training new model")
        predictor = InsurancePremiumPredictor()
        predictor.train('insurance.csv')
        predictor.save_model(MODEL_PATH)

if __name__ == '__main__':
    initialize_app()
    app.run(debug=True)