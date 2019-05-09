#import packages
import numpy as np 
import h5py

#load MNIST data
MNIST_data=h5py.File("MNISTdata.hdf5","r")
x_train=np.float32(MNIST_data["x_train"][:])
y_train=np.int32(np.array(MNIST_data["y_train"][:,0]))
x_test = np.float32( MNIST_data["x_test"][:] )
y_test = np.int32( np.array( MNIST_data["y_test"][:,0] ))

MNIST_data.close()

#uumber of hidden units, inputs and outputs
num_hidden=100
num_inputs = 28*28
num_outputs = 10

#define the softmax function
def softmax(z):
    return np.exp(z)/np.mat(np.sum(np.exp(z),axis=0))

#define sigmoid function 
def sigmoid(z):
    return np.exp(z)/(1+np.exp(z))

#initialize the weight
w1=np.random.randn(num_hidden,num_inputs) / np.sqrt(num_inputs)
w2=np.random.randn(num_outputs,num_hidden)/np.sqrt(num_inputs)

#initialize the bias
b1=np.mat(np.random.randn(num_hidden)/np.sqrt(num_hidden)).T
b2=np.mat(np.random.randn(num_outputs)/np.sqrt(num_outputs)).T

#stochastic gradient descent 
#train the model
for i in range(250000):
    index=np.random.choice(len(y_train),1)
    x=x_train[index,:]
    y=y_train[index]
    e_y=np.zeros((num_outputs,1))
    e_y[y,]=1
    z=np.dot(w1,x.T)+b1
    h=sigmoid(z)
    u=np.dot(w2,h)+b2
    f_x_theta=softmax(u)
    delu=-(e_y-f_x_theta)
    if i<150000:
        #gradient descent of b2
        b2=b2-0.03*delu
        #gradient descent of w2
        w2=w2-0.03*np.dot(delu,h.T)
        #sigma
        sigma=np.dot(w2.T,delu)
        sigma=np.array(sigma)
        h=np.array(h)
        #gradient descent of b1
        b1=b1-0.03*sigma*(1-h)*h
        #gradient descent of w1
        w1=w1-0.03*np.dot(sigma*(1-h)*h,x)
    if 150000<=i<170000:
        #gradient descent of b2
        b2=b2-0.003*delu
        #gradient descent of w2
        w2=w2-0.003*np.dot(delu,h.T)
        #sigma
        sigma=np.dot(w2.T,delu)
        sigma=np.array(sigma)
        h=np.array(h)
        #gradient descent of b1
        b1=b1-0.003*sigma*(1-h)*h
        #gradient descent of w1
        w1=w1-0.003*np.dot(sigma*(1-h)*h,x)
    if 170000<=i:
        #gradient descent of b2
        b2=b2-0.0003*delu
        #gradient descent of w2
        w2=w2-0.0003*np.dot(delu,h.T)
        #sigma
        sigma=np.dot(w2.T,delu)
        sigma=np.array(sigma)
        h=np.array(h)
        #gradient descent of b1
        b1=b1-0.0003*sigma*(1-h)*h
        #gradient descent of w1
        w1=w1-0.0003*np.dot(sigma*(1-h)*h,x)

#substitute the test dataset for the model
z=np.dot(w1,x_test.T)+b1
h=sigmoid(z)
u=np.dot(w2,h)+b2
output=softmax(u)

#the classification of the test data set
answer=[]
for i in range(len(y_test)):
    index=np.where(output[:,i]==max(output[:,i]))[0]
    answer.append(index)

#count the number of correctness classification
correct=0
for i,s in enumerate(answer):
    if s==y_test[i]:
        correct+=1
        
#test accuracy
correct/len(y_test)

