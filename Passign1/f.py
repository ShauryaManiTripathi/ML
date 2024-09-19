import random
from collections import Counter

# Create a random matrix of size 10 x 5
matrix = [[random.randint(0, 9) for _ in range(5)] for _ in range(10)]

print("Matrix:\n")
for row in matrix:
    print(row)

# Function to count the number of patterns with the same value for each feature
def count_patterns(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    
    # Initialize a list to hold the counts for each feature
    feature_counts = [None] * num_cols
    
    for col in range(num_cols):
        # Extract the column values
        column_values = [matrix[row][col] for row in range(num_rows)]
        # Count the frequency of each value in the column
        counts = Counter(column_values)
        feature_counts[col] = counts
    
    return feature_counts

# Get the counts of patterns for each feature
feature_counts = count_patterns(matrix)

# Display the results
for i, counts in enumerate(feature_counts):
    print(f"Feature {i + 1}:")
    for value, count in counts.items():
        print(f"  Value {value} occurs {count} times")
    print()
