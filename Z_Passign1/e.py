import numpy as np

#create random array of 10x5
matrix=np.random.randint(1,10,(10,5))
print(matrix)
print("--------------------------")


#create function to get min/max of specific axis
def minmax(matrix,axis,minormax):
    if minormax=="min":
        return np.min(matrix,axis)
    elif minormax=="max":
        return np.max(matrix,axis)
    else:
        return "Invalid input"

print(minmax(matrix,0,"min"))
print(minmax(matrix,0,"max"))
print(minmax(matrix,1,"min"))
print(minmax(matrix,1,"max"))
print("--------------------------")