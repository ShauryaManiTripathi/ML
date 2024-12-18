import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = pd.read_csv('iris.csv')
X = iris.iloc[:, 1:5].values  # Features
y = iris['Species'].values    # Labels

# K-means clustering implementation
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def kmeans(X, k, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)
        
        new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]
        
        if np.all([np.allclose(centroids[i], new_centroids[i]) for i in range(k)]):
            break
        
        centroids = new_centroids
    
    return centroids, clusters

# Perform K-means clustering
k = 3  # Number of clusters
centroids, clusters = kmeans(X, k)

# Select 25 samples closest to each centroid
def select_closest_samples(X, centroids, clusters, n_samples=25):
    selected_samples = []
    selected_labels = []
    
    for i, cluster in enumerate(clusters):
        distances = [euclidean_distance(point, centroids[i]) for point in cluster]
        sorted_indices = np.argsort(distances)[:n_samples]
        selected_samples.extend([cluster[j] for j in sorted_indices])
        
        # Find the corresponding labels
        for point in [cluster[j] for j in sorted_indices]:
            idx = np.where((X == point).all(axis=1))[0][0]
            selected_labels.append(y[idx])
    
    return np.array(selected_samples), np.array(selected_labels)

X_selected, y_selected = select_closest_samples(X, centroids, clusters)

# Split the selected samples into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_selected, y_selected, test_size=0.2, random_state=42)

# Use the remaining samples as the test set
mask = np.ones(len(X), dtype=bool)
for sample in X_selected:
    idx = np.where((X == sample).all(axis=1))[0][0]
    mask[idx] = False
X_test, y_test = X[mask], y[mask]

# Logistic Regression implementation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)
        
        # Initialize parameters
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        
        # One-hot encode the labels
        y_encoded = np.eye(n_classes)[np.searchsorted(self.classes, y)]
        
        # Gradient descent
        for _ in range(self.num_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = softmax(z)
            
            # Backward pass
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_encoded))
            db = (1 / n_samples) * np.sum(y_pred - y_encoded, axis=0)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = softmax(z)
        return self.classes[np.argmax(y_pred, axis=1)]

# Train the Logistic Regression model
model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X_train, y_train)

# Evaluate the model
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Predictions and accuracy on training set
y_train_pred = model.predict(X_train)
train_accuracy = accuracy(y_train, y_train_pred)

# Predictions and accuracy on validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy(y_val, y_val_pred)

# Predictions and accuracy on test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")