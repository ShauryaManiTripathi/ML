import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('salary_data.csv')

# Print dataset information
print("Dataset Information:")
print(f"Number of features: {df.shape[1] - 1}")  # Subtracting 1 because Salary is the target
print(f"Number of patterns: {df.shape[0]}")
print(f"Range of output (Salary): ${df['Salary'].min()} to ${df['Salary'].max()}")
print()

# Function to split the dataset
def split_dataset(df, test_size):
    X = df[['YearsExperience']]
    y = df['Salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    # Combine features and target for saving
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # Create filenames
    train_filename = f"{int(100 * (1-test_size)+1)}_{int(100 * test_size)}_train_data.csv"
    test_filename = f"{int(100 * (1-test_size)+1)}_{int(100 * test_size)}_test_data.csv"
    
    # Save to CSV files
    train_data.to_csv(train_filename, index=False)
    test_data.to_csv(test_filename, index=False)
    return X_train, X_test, y_train, y_test

# Perform splits
print("Dataset Splits:")
for test_size in np.arange(0.9, 0.0, -0.1):
    train_size = 1 - test_size
    X_train, X_test, y_train, y_test = split_dataset(df, test_size)
    print(f"Split ratio - Train: {train_size:.1f}, Test: {test_size:.1f}")
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    print()