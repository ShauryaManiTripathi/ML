import numpy as np

#create 4x5 matrix with values ranging from 1 to 10
matrix = np.random.randint(1,10,(4,5))

transposematrix=matrix.transpose()
print(matrix)
print(transposematrix)

#zero array of zeroes
zeroarray=np.zeros((10))
zero10=zeroarray*10
ones10=(zeroarray*10)+1
fives10=(zeroarray*10)+5
print(np.array([zero10,ones10,fives10]).flatten())
print(np.concatenate([zero10,ones10,fives10]))

#array of all even integers from 10 to 50
evenarray=np.arange(10,51,2)
print(evenarray)

#generate a random number between 0 and 1
randomnumber=np.random.rand(1)
print(randomnumber)

#save matrix into a file, then load is back
np.save('matrix.npy',matrix)
loadedmatrix=np.load('matrix.npy')
print(loadedmatrix)

#using np filehandling to save matrix
np.savetxt('matrix.txt',matrix)
loadedmatrix=np.loadtxt('matrix.txt')

#save in file without using numpy,after converting to string
file=open('matrix.txt','w')
file.write(str(matrix))
file.close()
file=open('matrix.txt','r')
loadedmatrix=file.read()
print(loadedmatrix)
print(type(loadedmatrix))
file.close()

#save in file without using numpy, use serializability
import pickle
file=open('matrix.txt','wb')
pickle.dump(matrix,file)
file.close()
file=open('matrix.txt','rb')
loadedmatrix=pickle.load(file)
print(loadedmatrix)
print(type(loadedmatrix))
file.close()

