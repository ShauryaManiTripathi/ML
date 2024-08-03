import numpy as np

# Create two random arrays
array1 = np.random.randint(0, 10, size=(3, 4))
array2 = np.random.randint(0, 10, size=(3, 4))

print("Array 1:\n", array1)
print("Array 2:\n", array2)

# i. Concatenate the two arrays
concat_arrays = np.concatenate((array1, array2), axis=0)
print("Concatenated Array:\n", concat_arrays)

# ii. Sort both arrays
sorted_array1 = np.sort(array1, axis=None)
sorted_array2 = np.sort(array2, axis=None)
print("Sorted Array 1:\n", sorted_array1.reshape(array1.shape))
print("Sorted Array 2:\n", sorted_array2.reshape(array2.shape))

# iii. Add the two arrays
added_arrays = np.add(array1, array2)
print("Added Arrays:\n", added_arrays)

# iv. Subtract the two arrays
subtracted_arrays = np.subtract(array1, array2)
print("Subtracted Arrays:\n", subtracted_arrays)

# v. Multiply the two arrays
multiplied_arrays = np.multiply(array1, array2)
print("Multiplied Arrays:\n", multiplied_arrays)

# vi. Divide the two arrays
divided_arrays = np.divide(array1, array2, where=(array2!=0))
print("Divided Arrays:\n", divided_arrays)
