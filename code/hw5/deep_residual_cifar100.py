print("==============================================================")
print("==============================================================")
print("==============================================================")
print("==============================================================")
print("Part1")
print("==============================================================")
print("==============================================================")
print("==============================================================")
print("==============================================================")
##Part1  
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
num_epochs = 30
learning_rate = 0.003
batch_size = 128
DIM=32

# Data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(DIM),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)

        if self.downsample is not None:
            residual = self.downsample(x)

        h += residual
        h=self.relu(h)
        return h

class ResNet(nn.Module):

    def __init__(self,block,layers):
        super(ResNet,self).__init__()
        self.in_channels = 32
        self.conv=nn.Conv2d(3,32,3,stride=1,padding=1,bias=False)
        self.bn=nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.02)
        self.block1=self.blocklayer(block,32,layers[0],stride=1)
        self.block2=self.blocklayer(block,64,layers[1],stride=2)
        self.block3=self.blocklayer(block,128,layers[2],stride=2)
        self.block4=self.blocklayer(block,256,layers[3],stride=2)
        self.pool=nn.MaxPool2d(kernel_size=4,stride=1)
        self.fc=nn.Linear(256,100)

    def blocklayer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,3, stride=stride, padding=1,bias=False),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self,x):
        h=self.conv(x)
        h=self.bn(h)
        h=self.relu(h)
        h=self.dropout(h)
        h=self.block1(h)
        h=self.block2(h)
        h=self.block3(h)
        h=self.block4(h)
        h=self.pool(h)
        h = h.view(h.size(0), -1)
        h=self.fc(h)
        return h 


model = ResNet(BasicBlock, [2, 4, 4, 2])

model.cuda()

criterion=nn.CrossEntropyLoss()

optimizer=optim.Adam(model.parameters(),lr=learning_rate)

train_loss = []
train_accu = []
test_accu = []


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


print("==============================================================")
print("==============================================================")
print("==============================================================")
print("==============================================================")
print("Part2")
print("==============================================================")
print("==============================================================")
print("==============================================================")
print("==============================================================")
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
num_epochs = 10
learning_rate = 0.003
batch_size = 128
no_of_hidden_units = 128

class FineTune(nn.Module):
    def __init__(self, resnet, num_classes):
        super(FineTune, self).__init__()
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs = resnet.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes)
        )
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def resnet18(model_urls, pretrained=True):
    """Load pre-trained ResNet-18 model in Pytorch."""
    model = torchvision.models.resnet.ResNet(
        torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])

    if pretrained:
        model.load_state_dict(torch.utils.model_zoo.load_url(
            model_urls, model_dir='../'))
        model = FineTune(model, num_classes=100)
    return model
  

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

model = resnet18("https://download.pytorch.org/models/resnet18-5c106cde.pth")
model.cuda()

criterion=nn.CrossEntropyLoss()

optimizer=optim.Adam(model.parameters(),lr=learning_rate)

train_loss = []
train_accu = []
test_accu = []

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