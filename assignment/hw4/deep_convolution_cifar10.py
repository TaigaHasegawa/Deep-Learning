# Load the packages 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

import math
import os
import time
import numpy as np

# Hyper-parameters
num_epochs = 100
learning_rate = 0.003
batch_size = 128
DIM = 32
no_of_hidden_units = 128


# Data augmentation
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(DIM, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(DIM),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

#dawonload the CIFAR10  and split it to train and test dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

#Deep convolution network 
class Convolution(nn.Module):

    def __init__(self,no_of_hidden_units):
        super(Convolution,self).__init__()
        self.conv1=nn.Conv2d(3,64,3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(64)
        self.conv2=nn.Conv2d(64,64,3,stride=1,padding=1)
        self.maxpool1= nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout1=nn.Dropout(p=0.5)
        self.conv3=nn.Conv2d(64,128,3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(128)
        self.conv4=nn.Conv2d(128,128,3,stride=1,padding=1)
        self.maxpool2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout2=nn.Dropout(p=0.5)
        self.conv5=nn.Conv2d(128,256,3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(256)
        self.conv6=nn.Conv2d(256,256,3,stride=1,padding=1)
        self.maxpool3=nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout3=nn.Dropout(p=0.5)
        self.conv7=nn.Conv2d(256,512,3,stride=1,padding=1)
        self.bn4=nn.BatchNorm2d(512)
        self.conv8=nn.Conv2d(512,512,3,stride=1,padding=1)
        self.maxpool4=nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout4=nn.Dropout(p=0.5)
        self.conv9=nn.Conv2d(512,512,3,stride=1,padding=1)
        self.bn5=nn.BatchNorm2d(512)
        self.conv10=nn.Conv2d(512,512,3,stride=1,padding=1)
        self.maxpool5=nn.MaxPool2d(kernel_size=2,stride=2)
        self.dropout5=nn.Dropout(p=0.5)
        self.fc1=nn.Linear(1*1*512,no_of_hidden_units)
        self.dropout6=nn.Dropout(p=0.5)
        self.fc2=nn.Linear(no_of_hidden_units,no_of_hidden_units)
        self.dropout7=nn.Dropout(p=0.5)
        self.fc3=nn.Linear(no_of_hidden_units,10)

    def forward(self,x):
        h=self.conv1(x)
        h=self.bn1(h)
        h=F.relu(h)
        h=self.conv2(h)
        h=self.dropout1(h)
        h=F.relu(h)
        h=self.maxpool1(h)
        h=self.conv3(h)
        h=self.bn2(h)
        h=F.relu(h)
        h=self.conv4(h)
        h=self.dropout2(h)
        h=F.relu(h)
        h=self.maxpool2(h)
        h=self.conv5(h)
        h=self.bn3(h)
        h=F.relu(h)
        h=self.conv6(h)
        h=self.dropout3(h)
        h=F.relu(h)
        h=self.maxpool3(h)
        h=self.conv7(h)
        h=self.bn4(h)
        h=self.conv8(h)
        h=self.dropout4(h)
        h=F.relu(h)
        h=self.maxpool4(h)
        h=self.conv9(h)
        h=self.bn5(h)
        h=self.conv10(h)
        h=self.dropout5(h)
        h=F.relu(h)
        h=self.maxpool5(h)
        h = h.reshape(h.size(0), -1)
        h=self.fc1(h)
        h=self.dropout6(h)
        h=F.relu(h)
        h=self.fc2(h)
        h=self.dropout7(h)
        h=F.relu(h)
        h=self.fc3(h)
        
        return(h) 


model=Convolution(no_of_hidden_units)
model.cuda()

criterion=nn.CrossEntropyLoss()

optimizer=optim.Adam(model.parameters(),lr=learning_rate)

train_loss = []
train_accu = []
test_accu = []

train_step = len(trainloader)
test_step=len(testloader)

# Train the model
for epoch in range(0,num_epochs):

    if(epoch%10==0):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    
    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    for batch_idx, data in enumerate(trainloader,0):
        
        X_train_batch, Y_train_batch=data

        if(Y_train_batch.shape[0]<batch_size):
            continue

        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()

        optimizer.zero_grad()
        h=model(X_train_batch)
        loss=criterion(h,Y_train_batch)
        pred=F.softmax(h,dim=1)

        loss.backward()
        optimizer.step()
        prediction=pred.data.max(1)[1]
        epoch_acc+=float(prediction.eq(Y_train_batch.data).sum())
        epoch_loss+=loss.item()
        epoch_counter+=batch_size
    
    epoch_acc/=epoch_counter
    epoch_loss /= (epoch_counter/batch_size)
    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    print(epoch, "Train:", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss)


    model.eval()

    epoch_acc=0.0
    epoch_loss=0.0

    epoch_counter=0


    for data in testloader:
        
        X_test_batch, Y_test_batch=data

        if(Y_test_batch.shape[0]<batch_size):
            continue

        X_test_batch = Variable(X_test_batch).cuda()
        Y_test_batch = Variable(Y_test_batch).cuda()

        h=model(X_test_batch)
        loss=criterion(h,Y_test_batch)
        pred=F.softmax(h,dim=1)
        prediction=pred.data.max(1)[1]
        epoch_acc+=float(prediction.eq(Y_test_batch.data).sum())
        epoch_loss+=loss.item()
        epoch_counter+=batch_size
    
    epoch_acc/=epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    test_accu.append(epoch_acc)
    
    print("TEST:  ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss)    

