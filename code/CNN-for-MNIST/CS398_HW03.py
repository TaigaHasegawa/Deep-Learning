#load the packages
import numpy as np 
import h5py

#load MNIST data
MNIST_data=h5py.File("MNISTdata.hdf5","r")
x_train=np.float32(MNIST_data["x_train"][:])
y_train=np.int32(np.array(MNIST_data["y_train"][:,0]))
x_test = np.float32( MNIST_data["x_test"][:] )
y_test = np.int32( np.array( MNIST_data["y_test"][:,0] ))

MNIST_data.close()

#transform the data into 2d 
x_train=x_train.reshape(60000,28,28)
x_test=x_test.reshape(10000,28,28)

#nbsumber of inputs, outputs and channels
num_inputs = 28
num_outputs = 10
num_channels=8

#define the softmax function
def softmax(z):
    return np.exp(z)/np.mat(np.sum(np.exp(z),axis=0))

#define the relu function
def relu(x):
    y = np.maximum(0, x)
    return y

#define the derivative of the relu function 
def relu_derivative(x):
    y = np.where( x > 0, 1, 0)
    return y

#initialize the filter
k=np.random.randn(3,3,num_channels)/ np.sqrt(num_channels)

#initialize the weight
w=np.random.randn(num_outputs,(num_inputs-2)*(num_inputs-2)*(num_channels))/ np.sqrt(num_inputs)

#initialize the bias
b=np.mat(np.random.randn(num_outputs)/np.sqrt(num_outputs)).T

#function calculating the feature maps 
def feature_maps_function(k,x,num_inputs,num_channels):
    feature_maps=np.zeros((num_inputs-2,num_inputs-2,num_channels))
    for p in range(num_channels):
        for i in range(num_inputs-2):
            for j in range(num_inputs-2):
                feature_maps[i,j,p]=sum(sum(k[0:2,0:2,p]*x[i:i+2,j:j+2]))
    return feature_maps

#function calculating delta k 
def delta_k(mul,x,num_channels,fil_size):
    deltak=np.zeros((fil_size,fil_size,num_channels))
    for p in range(num_channels):
        for i in range(fil_size):
            for j in range(fil_size):
                deltak[i,j,p]=sum(sum(mul[0:25,0:25,p]*x[i:i+25,j:j+25]))
    return deltak

#stochastic gradient descent 
for i in range(50000):
    #randonly select the one data 
    index=np.random.choice(len(y_train),1)
    x=x_train[index,:,:].reshape(num_inputs,num_inputs)
    y=y_train[index]
    #forward step
    feature_maps=feature_maps_function(k,x,num_inputs,num_channels)
    h=relu(feature_maps).reshape(5408,1)
    u=np.dot(w,h)+b
    f_x_theta=softmax(u)
    #backward step
    e_y=np.zeros((num_outputs,1))
    e_y[y,]=1
    delu=-(e_y-f_x_theta)
    delta=np.array(np.dot(delu.T,w)).reshape(26,26,num_channels)
    mul=relu_derivative(feature_maps)*delta
    deltak=delta_k(mul,x,num_channels,3)
    if i<10000:
        b=b-0.05*delu
        w=w-0.05*np.dot(delu,h.T)
        k=k-0.05*deltak
    if 10000<=i<30000:
        b=b-0.005*delu
        w=w-0.005*np.dot(delu,h.T)
        k=k-0.005*deltak
    if 30000<=i:
        b=b-0.0005*delu
        w=w-0.0005*np.dot(delu,h.T)
        k=k-0.0005*deltak

answer=[]
for ob in range(10000):
    #forward step of the test dataset
    feature_maps=feature_maps_function(k,x_test[ob,:,:],num_inputs,num_channels)
    h=relu(feature_maps).reshape(5408,1)
    u=np.dot(w,h)+b
    f_x_theta=softmax(u)
    #find the index where the probability is the max 
    index=np.where(f_x_theta==max(f_x_theta))[0]
    answer.append(index)

#count the number of correctness classification
correct=0
for i,s in enumerate(answer):
    if s==y_test[i]:
        correct+=1
        
#test accuracy
correct/len(y_test)
