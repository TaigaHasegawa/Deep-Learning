import numpy as np
import h5py
import time
import copy
from random import randint
#load MNIST data
MNIST_data = h5py.File("MNISTdata.hdf5", "r")
x_train = np.float32(MNIST_data["x_train"][:] )
y_train = np.int32(np.array(MNIST_data["y_train"][:,0]))
x_test = np.float32( MNIST_data["x_test"][:] )
y_test = np.int32( np.array( MNIST_data["y_test"][:,0] ) )
MNIST_data.close()

#number of inputs
num_inputs = 28*28
#number of outputs
num_outputs = 10

#softmax function to the matrix
def softmax_for_matrix(z):
    return np.exp(z)/np.sum(np.exp(z),axis=1).reshape(-1,1)

#softmax function to the vector
def softmax_for_vector(z):
    return np.exp(z)/np.sum(np.exp(z))

#initialize the theta
theta=np.random.randn(num_outputs,num_inputs) / np.sqrt(num_inputs)

#stochastic graadient descent 
for i in range(1,100000):
    e=np.zeros(10)
    index=np.random.choice(x_train.shape[0],1)
    x=x_train[index,]
    y=y_train[index,]
    score=np.dot(x,theta.T)
    e[y]=1
    if i<40000:
        theta=theta-0.6*-np.dot((e-softmax_for_vector(score)).reshape(-1,1),x.reshape(1,-1))
    if 40000<=i<70000:
        theta=theta-0.06*-np.dot((e-softmax_for_vector(score)).reshape(-1,1),x.reshape(1,-1))
    if 70000<=i:
        theta=theta-0.006*-np.dot((e-softmax_for_vector(score)).reshape(-1,1),x.reshape(1,-1))

#softmax function of the test data set
output=softmax_for_matrix(np.dot(x_test,theta.T))

#the classification of the test data set
answer=[]
for out in output:
    index=np.where(out==max(out))
    answer+=index

#count the number of correctness classification
correct=0
for i,s in enumerate(answer):
    if s==y_test[i]:
        correct+=1
        
#test accuracy
print(correct/len(y_test))