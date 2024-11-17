import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress the specific warning about recall
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

class ModelAnalyzer:
    def __init__(self, dataset_name, data_path):
        self.dataset_name = dataset_name
        self.data = pd.read_csv(data_path)
        self.X = self.data.iloc[:, :-1].values
        self.y = self.data.iloc[:, -1].values
        self.results_dict = {}
        
    def preprocess_data(self):
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        
    def evaluate_model(self, y_true, y_pred, model_name):
        precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, 
                                                                average=None, 
                                                                zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        unique_classes = np.unique(y_true)
        metrics = {}
        
        for i, cls in enumerate(unique_classes):
            metrics[f'Class_{cls}'] = {
                'Precision': precision[i],
                'Recall': recall[i]
            }
        metrics['Overall_Accuracy'] = accuracy
        
        return metrics
    
    def plot_learning_curves(self, train_scores, test_scores, param_values, param_name, model_name):
        try:
            plt.figure(figsize=(15, 5))
            
            # Plot 1: Training vs Testing Accuracy
            plt.subplot(1, 3, 1)
            plt.plot(param_values, train_scores, marker='o', label='Training Accuracy')
            plt.plot(param_values, test_scores, marker='o', label='Testing Accuracy')
            plt.xlabel(param_name)
            plt.ylabel('Accuracy')
            plt.title(f'{model_name}: Training vs Testing Accuracy')
            plt.grid(True)
            plt.legend()
            
            # Plot 2: Overfitting Analysis
            plt.subplot(1, 3, 2)
            plt.plot(param_values, np.array(train_scores) - np.array(test_scores),
                     marker='o', label='Overfitting Gap')
            plt.xlabel(param_name)
            plt.ylabel('Accuracy Difference')
            plt.title(f'{model_name}: Overfitting Analysis\n(Train - Test Accuracy)')
            plt.grid(True)
            
            # Plot 3: Cross-validation Scores
            plt.subplot(1, 3, 3)
            plt.plot(param_values, self.results_dict[model_name]['cv_scores'], 
                    marker='o', label='CV Score')
            plt.xlabel(param_name)
            plt.ylabel('CV Score')
            plt.title(f'{model_name}: 3-Fold Cross-validation Score')
            plt.grid(True)
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(f'{self.dataset_name}_{model_name}_analysis.png')
            plt.close()
            
        except Exception as e:
            print(f"Error creating plots for {model_name}: {str(e)}")
    
    def train_slp(self):
        print(f"\nTraining SLP for {self.dataset_name}...")
        # Hyperparameter tuning for SLP
        max_iters = [100, 500, 1000, 2000]
        train_scores = []
        test_scores = []
        cv_scores = []
        
        for iter_val in max_iters:
            slp = Perceptron(max_iter=iter_val, random_state=42)
            slp.fit(self.X_train_scaled, self.y_train)
            
            train_scores.append(accuracy_score(self.y_train, slp.predict(self.X_train_scaled)))
            test_scores.append(accuracy_score(self.y_test, slp.predict(self.X_test_scaled)))
            cv_scores.append(np.mean(cross_val_score(slp, self.X, self.y, cv=3)))
        
        # Store results
        self.results_dict['SLP'] = {
            'hyperparameters': {'max_iter': max_iters},
            'train_scores': train_scores,
            'test_scores': test_scores,
            'cv_scores': cv_scores
        }
        print("Hyperparameter tuning results for SLP:",self.dataset_name)
        print(self.results_dict['SLP'])
        
        # Train final model with best parameters
        best_iter = max_iters[np.argmax(cv_scores)]
        final_slp = Perceptron(max_iter=best_iter, random_state=42)
        final_slp.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate final model
        y_pred = final_slp.predict(self.X_test_scaled)
        self.results_dict['SLP']['metrics'] = self.evaluate_model(self.y_test, y_pred, 'SLP')
        
        # Plot learning curves
        self.plot_learning_curves(train_scores, test_scores, max_iters, 'Max Iterations', 'SLP')
    
    def train_mlp(self):
        print(f"Training MLP for {self.dataset_name}...")
        # Hyperparameter tuning for MLP
        hidden_layers = [(10,), (20,), (10, 10), (20, 10)]
        train_scores = []
        test_scores = []
        cv_scores = []
        
        for layers in hidden_layers:
            mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=1000, random_state=42)
            mlp.fit(self.X_train_scaled, self.y_train)
            
            train_scores.append(accuracy_score(self.y_train, mlp.predict(self.X_train_scaled)))
            test_scores.append(accuracy_score(self.y_test, mlp.predict(self.X_test_scaled)))
            cv_scores.append(np.mean(cross_val_score(mlp, self.X, self.y, cv=3)))
        
        # Store results
        self.results_dict['MLP'] = {
            'hyperparameters': {'hidden_layers': hidden_layers},
            'train_scores': train_scores,
            'test_scores': test_scores,
            'cv_scores': cv_scores
        }
        print("Hyperparameter tuning results for MLP:",self.dataset_name)
        print(self.results_dict['MLP'])
        
        # Train final model with best parameters
        best_layers = hidden_layers[np.argmax(cv_scores)]
        final_mlp = MLPClassifier(hidden_layer_sizes=best_layers, max_iter=1000, random_state=42)
        final_mlp.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate final model
        y_pred = final_mlp.predict(self.X_test_scaled)
        self.results_dict['MLP']['metrics'] = self.evaluate_model(self.y_test, y_pred, 'MLP')
        
        # Plot learning curves
        layer_names = [str(layer) for layer in hidden_layers]
        self.plot_learning_curves(train_scores, test_scores, range(len(hidden_layers)), 
                                'Hidden Layer Configuration', 'MLP')
    
    def train_knn(self):
        print(f"Training KNN for {self.dataset_name}...")
        # Hyperparameter tuning for KNN
        k_values = list(range(1, 16, 2))
        train_scores = []
        test_scores = []
        cv_scores = []
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train_scaled, self.y_train)
            
            train_scores.append(accuracy_score(self.y_train, knn.predict(self.X_train_scaled)))
            test_scores.append(accuracy_score(self.y_test, knn.predict(self.X_test_scaled)))
            cv_scores.append(np.mean(cross_val_score(knn, self.X, self.y, cv=3)))
        
        # Store results
        self.results_dict['KNN'] = {
            'hyperparameters': {'k_values': k_values},
            'train_scores': train_scores,
            'test_scores': test_scores,
            'cv_scores': cv_scores
        }
        print("Hyperparameter tuning results for KNN:",self.dataset_name)
        print(self.results_dict['KNN'])
        
        # Train final model with best parameters
        best_k = k_values[np.argmax(cv_scores)]
        final_knn = KNeighborsClassifier(n_neighbors=best_k)
        final_knn.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate final model
        y_pred = final_knn.predict(self.X_test_scaled)
        self.results_dict['KNN']['metrics'] = self.evaluate_model(self.y_test, y_pred, 'KNN')
        
        # Plot learning curves
        self.plot_learning_curves(train_scores, test_scores, k_values, 'k Value', 'KNN')
    
    def save_results(self):
        """Save results to Excel with proper DataFrame formatting"""
        try:
            with pd.ExcelWriter(f'{self.dataset_name}_results.xlsx', engine='openpyxl') as writer:
                # Create summary sheet
                summary_data = []
                for model_name in self.results_dict.keys():
                    summary_data.append({
                        'Model': model_name,
                        'Overall_Accuracy': self.results_dict[model_name]['metrics']['Overall_Accuracy'],
                        'Best_CV_Score': max(self.results_dict[model_name]['cv_scores'])
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Save detailed results for each model
                for model_name, results in self.results_dict.items():
                    # Hyperparameter tuning results
                    param_name = list(results['hyperparameters'].keys())[0]
                    param_values = results['hyperparameters'][param_name]
                    
                    if isinstance(param_values[0], tuple):
                        param_values = [str(val) for val in param_values]
                    
                    tuning_df = pd.DataFrame({
                        param_name: param_values,
                        'Training_Accuracy': results['train_scores'],
                        'Testing_Accuracy': results['test_scores'],
                        'CV_Score': results['cv_scores']
                    })
                    
                    sheet_name = f'{model_name}_Tuning'[:31]
                    tuning_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Metrics results
                    metrics_data = []
                    metrics_dict = results['metrics']
                    
                    for key, value in metrics_dict.items():
                        if key == 'Overall_Accuracy':
                            continue
                        if isinstance(value, dict):
                            metrics_data.append({
                                'Class': key,
                                'Precision': value['Precision'],
                                'Recall': value['Recall']
                            })
                    
                    if metrics_data:
                        metrics_df = pd.DataFrame(metrics_data)
                        sheet_name = f'{model_name}_Metrics'[:31]
                        metrics_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                print(f"Results saved successfully for {self.dataset_name}")
                
        except Exception as e:
            print(f"Error saving results for {self.dataset_name}: {str(e)}")
            raise

def analyze_dataset(dataset_name, data_path):
    print(f"\nAnalyzing dataset: {dataset_name}")
    try:
        analyzer = ModelAnalyzer(dataset_name, data_path)
        analyzer.preprocess_data()
        
        # Train all models
        analyzer.train_slp()
        analyzer.train_mlp()
        analyzer.train_knn()
        
        # Save results
        print(f"Saving results for {dataset_name}...")
        analyzer.save_results()
        
        return analyzer.results_dict
        
    except Exception as e:
        print(f"Error analyzing {dataset_name} dataset: {str(e)}")
        return None

# Main execution
if __name__ == "__main__":
    # Define datasets
    datasets = {
        'iris': 'iris.csv',  # Update with your actual path
        'wine': 'wine.csv'   # Update with your actual path
    }

    # Analyze all datasets
    results = {}
    for name, path in datasets.items():
        try:
            dataset_results = analyze_dataset(name, path)
            if dataset_results:
                results[name] = dataset_results
                print(f"Successfully analyzed {name} dataset")
            else:
                print(f"Failed to analyze {name} dataset")
        except Exception as e:
            print(f"Critical error analyzing {name} dataset: {str(e)}")
    
    print("\nAnalysis complete. Check the generated Excel files and plots for results.")