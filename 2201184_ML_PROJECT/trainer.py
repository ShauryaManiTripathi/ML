import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import logging
from datetime import datetime
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Set up logging
log_dir = 'logs'
results_dir = 'results'
plots_dir = 'plots'
for directory in [log_dir, results_dir, plots_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def create_enhanced_visualizations(data):
    """Create and save enhanced visualizations including pairplot with hue and correlation heatmap."""
    logging.info("Creating enhanced visualizations")
    
    # Set the style using seaborn's default
    sns.set_theme()
    
    # Create pairplot with hue
    logging.info("Creating pairplot with hue")
    numerical_cols = ['age', 'bmi', 'children', 'charges']
    plot_data = data[numerical_cols + ['smoker']].copy()
    
    # Create pairplot with improved aesthetics
    pairplot = sns.pairplot(
        data=plot_data,
        hue='smoker',
        diag_kind='kde',
        plot_kws={'alpha': 0.6},
        diag_kws={'alpha': 0.7},
        palette='husl'
    )
    
    # Enhance the pairplot
    pairplot.fig.suptitle('Feature Relationships by Smoking Status', y=1.02, fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout()
    pairplot.savefig(f'{plots_dir}/enhanced_pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create correlation heatmap
    logging.info("Creating correlation heatmap")
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation matrix
    correlation_matrix = data[numerical_cols].corr()
    
    # Create heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        fmt='.2f'
    )
    
    plt.title('Feature Correlation Heatmap', pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create violin plots for charges distribution
    logging.info("Creating violin plots")
    plt.figure(figsize=(12, 6))
    
    # Violin plot for charges by smoking status
    sns.violinplot(
        data=data,
        x='smoker',
        y='charges',
        palette='husl'
    )
    
    plt.title('Distribution of Insurance Charges by Smoking Status', pad=20, fontsize=14)
    plt.xlabel('Smoking Status', fontsize=12)
    plt.ylabel('Insurance Charges ($)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/charges_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_vs_actual(y_true, y_pred, title):
    """Create and save an enhanced scatter plot of predicted vs actual values."""
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with hexbin for better density visualization
    plt.hexbin(y_true, y_pred, gridsize=30, cmap='YlOrRd', mincnt=1)
    plt.colorbar(label='Count')
    
    # Add diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'b--', label='Perfect Prediction', linewidth=2)
    
    plt.xlabel('Actual Insurance Charges ($)', fontsize=12)
    plt.ylabel('Predicted Insurance Charges ($)', fontsize=12)
    plt.title(title, pad=20, fontsize=14)
    
    # Add metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    stats_text = f'R² = {r2:.3f}\nRMSE = ${rmse:,.2f}\nMAE = ${mae:,.2f}'
    plt.text(0.05, 0.95, stats_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top',
             fontsize=10)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/prediction_vs_actual_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def load_and_preprocess_data(file_path):
    """Load and preprocess the insurance dataset."""
    logging.info("Loading dataset from %s", file_path)
    data = pd.read_csv(file_path)
    
    logging.info("Starting preprocessing pipeline")
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    data['sex'] = le_sex.fit_transform(data['sex'])
    data['smoker'] = le_smoker.fit_transform(data['smoker'])
    
    # One-hot encode region
    data = pd.get_dummies(data, columns=['region'], drop_first=True)
    
    # Save encoders for later use
    with open('encoders.pkl', 'wb') as f:
        pickle.dump({'sex': le_sex, 'smoker': le_smoker}, f)
    
    return data

def evaluate_fold(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance on a single fold."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    return metrics, y_pred

def perform_hyperparameter_tuning(X, y, kfold):
    """Perform hyperparameter tuning for multiple models with detailed reporting."""
    models = {
        'Linear Regression': (LinearRegression(), {}),
        'Ridge': (Ridge(), {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr']
        }),
        'Lasso': (Lasso(), {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'selection': ['cyclic', 'random']
        }),
        'ElasticNet': (ElasticNet(), {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'selection': ['cyclic', 'random']
        })
    }
    
    results = {}
    best_model = None
    best_score = float('inf')
    best_scaler = None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    for model_name, (model, params) in models.items():
        logging.info(f"\nEvaluating {model_name}")
        model_results = {
            'params_performance': [],
            'fold_performance': [],
            'best_params': None,
            'best_metrics': None
        }
        
        # If there are hyperparameters to tune
        if params:
            from sklearn.model_selection import ParameterGrid
            param_grid = ParameterGrid(params)
            
            for param_set in param_grid:
                param_performance = {
                    'parameters': param_set,
                    'fold_results': [],
                    'average_metrics': None
                }
                
                # Initialize model with current parameters
                current_model = model.__class__(**param_set)
                fold_metrics = []
                
                # Perform k-fold cross-validation
                for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_scaled)):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    metrics, _ = evaluate_fold(current_model, X_train, X_test, y_train, y_test)
                    metrics['fold'] = fold_idx + 1
                    fold_metrics.append(metrics)
                    param_performance['fold_results'].append(metrics)
                
                # Calculate average metrics across folds
                avg_metrics = {
                    'mse': np.mean([m['mse'] for m in fold_metrics]),
                    'rmse': np.mean([m['rmse'] for m in fold_metrics]),
                    'mae': np.mean([m['mae'] for m in fold_metrics]),
                    'r2': np.mean([m['r2'] for m in fold_metrics])
                }
                param_performance['average_metrics'] = avg_metrics
                model_results['params_performance'].append(param_performance)
                
                # Update best parameters if current is better
                if avg_metrics['mse'] < best_score:
                    best_score = avg_metrics['mse']
                    best_model = current_model
                    best_scaler = scaler
                    model_results['best_params'] = param_set
                    model_results['best_metrics'] = avg_metrics
        
        else:  # For Linear Regression with no hyperparameters
            fold_metrics = []
            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_scaled)):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                metrics, _ = evaluate_fold(model, X_train, X_test, y_train, y_test)
                metrics['fold'] = fold_idx + 1
                fold_metrics.append(metrics)
                model_results['fold_performance'].append(metrics)
            
            avg_metrics = {
                'mse': np.mean([m['mse'] for m in fold_metrics]),
                'rmse': np.mean([m['rmse'] for m in fold_metrics]),
                'mae': np.mean([m['mae'] for m in fold_metrics]),
                'r2': np.mean([m['r2'] for m in fold_metrics])
            }
            model_results['best_metrics'] = avg_metrics
            
            if avg_metrics['mse'] < best_score:
                best_score = avg_metrics['mse']
                best_model = model
                best_scaler = scaler
        
        results[model_name] = model_results
    
    return results, best_model, best_scaler

def save_detailed_results(results, feature_names):
    """Save detailed results including fold-wise and parameter-wise performance."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'{results_dir}/detailed_model_results_{timestamp}.txt'
    
    with open(results_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("DETAILED INSURANCE PREMIUM PREDICTION MODEL EVALUATION RESULTS\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n\n")
        
        # Find best overall model
        best_model_name = min(results.keys(), 
                            key=lambda k: results[k]['best_metrics']['mse'] if results[k]['best_metrics'] else float('inf'))
        
        f.write("BEST OVERALL MODEL SUMMARY\n")
        f.write("-" * 50 + "\n")
        f.write(f"Best performing model: {best_model_name}\n")
        if results[best_model_name]['best_params']:
            f.write(f"Best parameters: {json.dumps(results[best_model_name]['best_params'], indent=2)}\n")
        f.write("\nBest model metrics:\n")
        metrics = results[best_model_name]['best_metrics']
        f.write(f"MSE:  {metrics['mse']:.2f}\n")
        f.write(f"RMSE: {metrics['rmse']:.2f}\n")
        f.write(f"MAE:  {metrics['mae']:.2f}\n")
        f.write(f"R2:   {metrics['r2']:.4f}\n\n")
        
        # Detailed results for each model
        for model_name, model_results in results.items():
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"MODEL: {model_name}\n")
            f.write("=" * 100 + "\n\n")
            
            # If model has hyperparameters
            if model_results['params_performance']:
                f.write("HYPERPARAMETER PERFORMANCE\n")
                f.write("-" * 50 + "\n\n")
                
                for param_perf in model_results['params_performance']:
                    f.write(f"Parameters: {json.dumps(param_perf['parameters'], indent=2)}\n")
                    f.write("\nFold-wise performance:\n")
                    
                    # Create a DataFrame for better formatting
                    fold_df = pd.DataFrame(param_perf['fold_results'])
                    f.write(fold_df.to_string() + "\n\n")
                    
                    f.write("Average metrics across folds:\n")
                    avg_metrics = param_perf['average_metrics']
                    f.write(f"MSE:  {avg_metrics['mse']:.2f}\n")
                    f.write(f"RMSE: {avg_metrics['rmse']:.2f}\n")
                    f.write(f"MAE:  {avg_metrics['mae']:.2f}\n")
                    f.write(f"R2:   {avg_metrics['r2']:.4f}\n")
                    f.write("\n" + "-" * 50 + "\n\n")
            
            # For models without hyperparameters (Linear Regression)
            else:
                f.write("FOLD-WISE PERFORMANCE\n")
                f.write("-" * 50 + "\n\n")
                
                fold_df = pd.DataFrame(model_results['fold_performance'])
                f.write(fold_df.to_string() + "\n\n")
                
                f.write("Average metrics across folds:\n")
                avg_metrics = model_results['best_metrics']
                f.write(f"MSE:  {avg_metrics['mse']:.2f}\n")
                f.write(f"RMSE: {avg_metrics['rmse']:.2f}\n")
                f.write(f"MAE:  {avg_metrics['mae']:.2f}\n")
                f.write(f"R2:   {avg_metrics['r2']:.4f}\n")
                f.write("\n" + "-" * 50 + "\n\n")

def save_best_model_summary(results, feature_names, timestamp=None):
    """
    Create a dedicated summary file for the best performing model with detailed analysis.
    
    Args:
        results: Dictionary containing all model results
        feature_names: List of feature names used in the model
        timestamp: Optional timestamp for file naming
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_file = f'{results_dir}/best_model_summary_{timestamp}.txt'
    
    # Find the best model
    best_model_name = min(results.keys(), 
                         key=lambda k: results[k]['best_metrics']['mse'] if results[k]['best_metrics'] else float('inf'))
    best_model_results = results[best_model_name]
    
    with open(summary_file, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("BEST MODEL SUMMARY REPORT\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Model Overview
        f.write("MODEL OVERVIEW\n")
        f.write("-" * 50 + "\n")
        f.write(f"Best Performing Model: {best_model_name}\n")
        if best_model_results['best_params']:
            f.write("\nOptimal Hyperparameters:\n")
            f.write(json.dumps(best_model_results['best_params'], indent=2))
        f.write("\n\n")
        
        # Performance Metrics
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 50 + "\n")
        metrics = best_model_results['best_metrics']
        f.write("Overall Metrics:\n")
        f.write(f"Mean Squared Error (MSE):     {metrics['mse']:.4f}\n")
        f.write(f"Root Mean Squared Error:       {metrics['rmse']:.4f}\n")
        f.write(f"Mean Absolute Error (MAE):     {metrics['mae']:.4f}\n")
        f.write(f"R-squared Score (R²):         {metrics['r2']:.4f}\n\n")
        
        # Fold-wise Performance
        f.write("CROSS-VALIDATION PERFORMANCE\n")
        f.write("-" * 50 + "\n")
        if best_model_results['params_performance']:
            # For models with hyperparameters
            best_param_perf = min(best_model_results['params_performance'],
                                key=lambda x: x['average_metrics']['mse'])
            
            f.write("Fold-wise Results for Best Parameters:\n")
            fold_df = pd.DataFrame(best_param_perf['fold_results'])
            f.write("\n" + fold_df.to_string() + "\n\n")
            
            # Calculate standard deviations
            std_metrics = {
                'mse': np.std([fold['mse'] for fold in best_param_perf['fold_results']]),
                'rmse': np.std([fold['rmse'] for fold in best_param_perf['fold_results']]),
                'mae': np.std([fold['mae'] for fold in best_param_perf['fold_results']]),
                'r2': np.std([fold['r2'] for fold in best_param_perf['fold_results']])
            }
            
            f.write("\nMetric Stability (Standard Deviations):\n")
            f.write(f"MSE Standard Deviation:    {std_metrics['mse']:.4f}\n")
            f.write(f"RMSE Standard Deviation:   {std_metrics['rmse']:.4f}\n")
            f.write(f"MAE Standard Deviation:    {std_metrics['mae']:.4f}\n")
            f.write(f"R² Standard Deviation:     {std_metrics['r2']:.4f}\n")
        else:
            # For models without hyperparameters (e.g., Linear Regression)
            fold_df = pd.DataFrame(best_model_results['fold_performance'])
            f.write("\n" + fold_df.to_string() + "\n\n")
            
            # Calculate standard deviations
            std_metrics = {
                'mse': np.std([fold['mse'] for fold in best_model_results['fold_performance']]),
                'rmse': np.std([fold['rmse'] for fold in best_model_results['fold_performance']]),
                'mae': np.std([fold['mae'] for fold in best_model_results['fold_performance']]),
                'r2': np.std([fold['r2'] for fold in best_model_results['fold_performance']])
            }
            
            f.write("\nMetric Stability (Standard Deviations):\n")
            f.write(f"MSE Standard Deviation:    {std_metrics['mse']:.4f}\n")
            f.write(f"RMSE Standard Deviation:   {std_metrics['rmse']:.4f}\n")
            f.write(f"MAE Standard Deviation:    {std_metrics['mae']:.4f}\n")
            f.write(f"R² Standard Deviation:     {std_metrics['r2']:.4f}\n")
        
        # Model Information
        f.write("\nMODEL INFORMATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"Number of Features: {len(feature_names)}\n")
        f.write("\nFeatures Used:\n")
        for i, feature in enumerate(feature_names, 1):
            f.write(f"{i}. {feature}\n")
        
        # Recommendations and Notes
        f.write("\nRECOMMENDATIONS AND NOTES\n")
        f.write("-" * 50 + "\n")
        f.write("1. Model Usage:\n")
        f.write("   - The model requires standardized input features\n")
        f.write("   - Use the saved scaler before making predictions\n")
        if best_model_results['best_params']:
            f.write("\n2. Hyperparameter Insights:\n")
            for param, value in best_model_results['best_params'].items():
                f.write(f"   - {param}: {value}\n")
        f.write("\n3. Performance Characteristics:\n")
        f.write(f"   - The model explains {metrics['r2']*100:.2f}% of the variance in the target variable\n")
        f.write(f"   - Average prediction error (MAE): {metrics['mae']:.2f} units\n")
        
        # Date and Version Information
        f.write("\nMODEL VERSION INFORMATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model File: best_linear_model.pkl\n")
        f.write(f"Feature Columns File: feature_columns.pkl\n")
        f.write(f"Encoders File: encoders.pkl\n")

def main():
    # Load and preprocess data
    data = load_and_preprocess_data('insurance.csv')
    
    # Create enhanced visualizations before preprocessing
    create_enhanced_visualizations(data)
    
    # Prepare features and target
    X = data.drop(['charges'], axis=1)
    y = data['charges']
    
    # Set up KFold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Get timestamp for consistent file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Perform hyperparameter tuning with detailed k-fold validation
    results, best_model, best_scaler = perform_hyperparameter_tuning(X, y, kfold)
    
    # Create enhanced prediction vs actual plot using the best model
    X_scaled = best_scaler.transform(X)
    y_pred = best_model.predict(X_scaled)
    plot_prediction_vs_actual(y, y_pred, 'Best Model: Predicted vs Actual Insurance Charges')
    
    # Save other results as before
    save_detailed_results(results, X.columns)
    save_best_model_summary(results, list(X.columns), timestamp)
    
    # Save best model and scaler
    with open('best_linear_model.pkl', 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': best_scaler}, f)
    
    # Save feature columns
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)

if __name__ == "__main__":
    main()