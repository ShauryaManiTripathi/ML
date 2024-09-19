def dimensionof(matrix):
    if(type(matrix)==list):
        return 1+dimensionof(matrix[0])
    else:
        return 0

def shapeof(matrix):
    if(type(matrix)==list):
        return [len(matrix)]+shapeof(matrix[0])
    else:
        return []

def sizeof(matrix):
    if(type(matrix)==list):
        return len(matrix)*sizeof(matrix[0])
    else:
        return 1

matrix=[[1,6,7,9],[7,9,3,5]]
print(dimensionof(matrix))
print(shapeof(matrix))
print(sizeof(matrix))