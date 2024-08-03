import random

# Create two random 2D arrays
def create_random_array(rows, cols):
    return [[random.randint(0, 9) for _ in range(cols)] for _ in range(rows)]

array1 = create_random_array(3, 4)
array2 = create_random_array(3, 4)

print("Array 1:\n", array1)
print("Array 2:\n", array2)

# i. Concatenate the two arrays
concat_arrays = array1 + array2
print("Concatenated Array:\n", concat_arrays)

# ii. Sort both arrays
sorted_array1 = [sorted(row) for row in array1]
sorted_array2 = [sorted(row) for row in array2]
print("Sorted Array 1:\n", sorted_array1)
print("Sorted Array 2:\n", sorted_array2)

# iii. Add the two arrays
added_arrays = [[array1[i][j] + array2[i][j] for j in range(len(array1[0]))] for i in range(len(array1))]
print("Added Arrays:\n", added_arrays)

# iv. Subtract the two arrays
subtracted_arrays = [[array1[i][j] - array2[i][j] for j in range(len(array1[0]))] for i in range(len(array1))]
print("Subtracted Arrays:\n", subtracted_arrays)

# v. Multiply the two arrays
multiplied_arrays = [[array1[i][j] * array2[i][j] for j in range(len(array1[0]))] for i in range(len(array1))]
print("Multiplied Arrays:\n", multiplied_arrays)

# vi. Divide the two arrays
divided_arrays = [[array1[i][j] / array2[i][j] if array2[i][j] != 0 else float('inf') for j in range(len(array1[0]))] for i in range(len(array1))]
print("Divided Arrays:\n", divided_arrays)
