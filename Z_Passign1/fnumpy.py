import numpy as np
from collections import Counter

matrix=np.random.randint(1,10,(10,5))

print(matrix)

#get number of patterns having same ith feature for each i

def count_patterns(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    feature_counts = [None] * num_cols
    for col in range(num_cols):
        column_values = [matrix[row][col] for row in range(num_rows)]
        counts = Counter(column_values)
        feature_counts[col] = counts
    return feature_counts

#function that doesnt uses counter
def count_patterns_no_counter(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    feature_counts = [None] * num_cols
    for col in range(num_cols):
        column_values = [matrix[row][col] for row in range(num_rows)]
        counts = {}
        for val in column_values:
            if val in counts:
                counts[val]+=1
            else:
                counts[val]=1
        feature_counts[col] = counts
    return feature_counts


print(list(count_patterns(matrix)))
print('------------------')
print(count_patterns_no_counter(matrix))