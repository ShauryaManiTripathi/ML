import numpy as np
from sklearn.datasets import load_iris


iris = load_iris()
x = iris.data
y = iris.target

x = (x - x.mean(axis=0)) / x.std(axis=0)
#for k-fold
n_folds = 5
learning_rates = [0.01, 0.001, 0.0001]
epochs_list = [100, 200, 300]

#store results
all_metrics = []
best_accuracy = 0
best_params = {}

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

def custom_kfold(x, y, n_splits=5, shuffle=True, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(x)
    fold_size = n_samples // n_splits
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    folds = []
    for i in range(n_splits):
        start = i * fold_size
        end = start + fold_size if i < n_splits - 1 else n_samples
        val_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_indices, val_indices))
    
    return folds


def one_hot_encode(y):
    unique_classes = np.unique(y)
    encoded = np.zeros((len(y), len(unique_classes)))
    for i, label in enumerate(y):
        encoded[i, label] = 1
    return encoded

class singlelayerperceptron:
    def __init__(self, input_size, num_classes, learning_rate=0.01, epochs=100):
        self.weights = np.random.randn(input_size, num_classes) * 0.01
        self.bias = np.zeros((1, num_classes))
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.training_loss = []
        
    def forward(self, x):
        self.z = np.dot(x, self.weights) + self.bias
        return sigmoid(self.z)
    
    def compute_loss(self, y_pred, y_true):
        return -np.mean(np.sum(y_true * np.log(np.clip(y_pred, 1e-10, 1.0)) + 
                              (1 - y_true) * np.log(np.clip(1 - y_pred, 1e-10, 1.0)), axis=1))
    
    def train(self, x, y, x_val=None, y_val=None):
        train_losses = []
        val_losses = []
        n_samples = x.shape[0]
        
        for epoch in range(self.epochs):
            y_pred = self.forward(x)
            
            error = y_pred - y
            dw = np.dot(x.T, error * sigmoid_derivative(self.z)) / n_samples
            db = np.mean(error * sigmoid_derivative(self.z), axis=0, keepdims=True)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            train_loss = self.compute_loss(y_pred, y)
            train_losses.append(train_loss)
            
            if x_val is not None and y_val is not None:
                val_pred = self.forward(x_val)
                val_loss = self.compute_loss(val_pred, y_val)
                val_losses.append(val_loss)
        
        return train_losses, val_losses
    
    def predict(self, x):
        probabilities = self.forward(x)
        return np.argmax(probabilities, axis=1)

def compute_metrics(y_true, y_pred):
    classes = np.unique(y_true)
    metrics = {}
    
    for c in classes:
        true_positive = np.sum((y_true == c) & (y_pred == c))
        false_positive = np.sum((y_true != c) & (y_pred == c))
        false_negative = np.sum((y_true == c) & (y_pred != c))
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        
        metrics[f'class_{c}'] = {
            'precision': precision,
            'recall': recall
        }
    
    overall_accuracy = np.mean(y_true == y_pred)
    metrics['overall_accuracy'] = overall_accuracy
    
    return metrics

folds = custom_kfold(x, y, n_splits=n_folds, shuffle=True, random_state=42)


for lr in learning_rates:
    for epochs in epochs_list:
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            x_train, x_val = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            y_train_encoded = one_hot_encode(y_train)
            y_val_encoded = one_hot_encode(y_val)
            
            model = singlelayerperceptron(
                input_size=x.shape[1],
                num_classes=len(np.unique(y)),
                learning_rate=lr,
                epochs=epochs
            )
            
            train_losses, val_losses = model.train(x_train, y_train_encoded, x_val, y_val_encoded)
            
            y_pred = model.predict(x_val)
            
            metrics = compute_metrics(y_val, y_pred)
            metrics['learning_rate'] = lr
            metrics['epochs'] = epochs
            fold_metrics.append(metrics)
            
            if metrics['overall_accuracy'] > best_accuracy:
                best_accuracy = metrics['overall_accuracy']
                best_params = {'learning_rate': lr, 'epochs': epochs}
        
        avg_metrics = {
            'learning_rate': lr,
            'epochs': epochs,
            'overall_accuracy': np.mean([m['overall_accuracy'] for m in fold_metrics])
        }
        
        for class_idx in range(3):
            avg_metrics[f'class_{class_idx}_precision'] = np.mean([m[f'class_{class_idx}']['precision'] for m in fold_metrics])
            avg_metrics[f'class_{class_idx}_recall'] = np.mean([m[f'class_{class_idx}']['recall'] for m in fold_metrics])
        
        all_metrics.append(avg_metrics)
        
        print(f"\nresults for lr={lr}, epochs={epochs}:")
        print(f"average accuracy across folds: {avg_metrics['overall_accuracy']:.4f}")
        for i in range(3):
            print(f"class {i}:")
            print(f"  precision: {avg_metrics[f'class_{i}_precision']:.4f}")
            print(f"  recall: {avg_metrics[f'class_{i}_recall']:.4f}")

print("\nfinal results:")
print(f"best hyperparameters: learning rate = {best_params['learning_rate']}, epochs = {best_params['epochs']}")
print(f"best accuracy: {best_accuracy:.4f}")