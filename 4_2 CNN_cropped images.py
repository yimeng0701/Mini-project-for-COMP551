# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 20:50:25 2018

@author: Ibtihel, Sunanda, Yimeng
"""

#original image+early stopping +

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchvision import transforms as tfs

#Hyper Parameters
EPOCH=80
BATCH_SIZE=80
LR=0.001
Valid_Bath_Size=1000
TRY=0


#loading data
if TRY :
    train_x=np.zeros((10000,28*28))
else: train_x=np.zeros((50000,28*28))

i=0
#plz change the name of the file accordingly; correspond to the ProcessedData/LargestDigit/train_x.csv 
filename1="cs.mcgill.ca/~ksinha4/datasets/kaggle/train_x.csv"
with open(filename1,'r',encoding='utf8') as f:
    for line in f.readlines():
       # print(i)
        img=np.array(line.split(','))
        img=img.astype(np.float)
       # for k in range(64*64):
       #     if img[k]<240:  img[k]=0
       # img=img/255
        train_x[i]=img
        i+=1
        if TRY:
            if i>=10000: break
    f.close()

train_y=np.loadtxt("cs.mcgill.ca/~ksinha4/datasets/kaggle/train_y.csv",dtype=int,delimiter=',')
if TRY:  train_y=train_y[:10000]


# shuffle and split of training data
train_y=train_y.reshape(-1,1)
train_x=np.hstack((train_x,train_y))     #train_data=train_x+train_y       data+label
np.random.shuffle(train_x)
train_data, valid_data= train_test_split(train_x, test_size=0.1,random_state=4)
valid_num=valid_data.shape[0]

# transform training data from numpy to tensor
train_x=train_data[:,:-1].reshape(-1,28,28)
train_x=torch.from_numpy(train_x)     #transfo.type(torch.FloatTensor)rm to tensor from
train_x = torch.unsqueeze(train_x, dim=1).type(torch.FloatTensor)
train_num=train_x.shape[0]
train_y = train_data[:, -1]
train_y = torch.from_numpy(train_y).type(torch.LongTensor)              

dataset = Data.TensorDataset(train_x, train_y)

# transform validation data from numpy to tensor
valid_x=valid_data[:,:-1].reshape(-1,28,28)
valid_x=torch.from_numpy(valid_x).type(torch.FloatTensor)     #transform to tensor from
valid_x=Variable(torch.unsqueeze(valid_x,dim=1),volatile=True).cuda()

valid_y=valid_data[:,-1]
valid_y=torch.from_numpy(valid_y).type(torch.LongTensor).cuda()         

del train_data
del valid_data

dataset=Data.TensorDataset(train_x,train_y)
#train_loader=Data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)

#construct CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1=nn.Sequential(   #input shape(1,64,64)
            nn.Conv2d(
                in_channels=1,      #input depth
                out_channels=20,    #output depth
                kernel_size=5,      #filter size
                stride=1,           #filter movement step
                padding=2         #padding=(kernel-stride)/2
            ),  #output shape(20,64,64)
            nn.BatchNorm2d(20),
            nn.ReLU(),  #activation
            nn.Conv2d(20,40,5,1,2),
            nn.BatchNorm2d(40),
            nn.ReLU(),  # activation
         #   nn.MaxPool2d(2),    #output shape(40,32,32)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(40,80,5,1,2) , #output shape(80,32,32)
            nn.BatchNorm2d(80),
            nn.ReLU(),
            
         #   nn.MaxPool2d(2),    #output shape(80,16,16)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(80, 160, 5,1,2),  # output shape(160,16,16)
            nn.BatchNorm2d(160),
            nn.ReLU(),
            
            nn.MaxPool2d(2),  # output shape(160,8,8)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(160, 320, 5, 1, 2),  # output shape(320,8,8)
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.MaxPool2d(2),  # output shape(320,4,4)
        )
        self.fnet4=nn.Sequential(
            nn.Linear(320*7*7,1000),      #fully connected layer, output:1000
            nn.BatchNorm2d(1000),
            nn.ReLU(),
        )
        
        self.dropout=nn.Dropout2d(p=0.4)  #0.4 dropout
        self.out=nn.Linear(1000,10)       #fully cnnected layer, output 10 classes

    def forward(self, x):
        x=self.conv1(x)
        # x=self.dropout(x)
        x=self.conv2(x)
        # x = self.dropout(x)
        x=self.conv3(x)
        x = self.conv4(x)
        x = self.dropout(x)
        x=x.view(x.size(0),-1)      #convert to (batch_size,32*7*7)
        x = self.fnet4(x)
        output=self.out(x)
        return output
cnn=CNN()
cnn.cuda()      #GPU
print(cnn)
optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)   #optimize all cnn parameters
loss_func=nn.CrossEntropyLoss()


best_validation_accuracy = 0.0    # the best validation accuracy in this time
last_improvement = 0     # iteration that has a improvment
require_improvement = 1200    # stop if no improvment in 1200 iteration
total_iterations = 0
break_signal=0
b_x=torch.zeros((BATCH_SIZE,1,28,28))

"""""
# data agumention     randomly choose immages to do following transformation
transform = tfs.Compose([         
    tfs.ToPILImage(),  
    tfs.RandomRotation(180,expand=True),       #rotation
    tfs.RandomResizedCrop(28,scale=(0.8,1)),   #resize to 64*64
    tfs.RandomHorizontalFlip(),                #flip horizontal        
    tfs.ToTensor()]
)
"""""

#training and testing
t=0
accs=[]
recordloss=[]
totalloss=[]
count=[]
for epoch in range(EPOCH):
    train_loader = Data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
    for step,(x,y) in enumerate(train_loader):
      #  for i in range (BATCH_SIZE):
      #      b_x[i]=transform(x[i])
        b_x=Variable(x).cuda()     #GPU
        b_y=Variable(y).cuda()      #GPU
        total_iterations+=1         #compute iteration times
        output=cnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()   #clear gradients for this training step
        loss.backward()         #backpropagation,compute gradients
        totalloss.append(loss.data[0])
        optimizer.step()        #apply gradients
        if step%50==0:
            for m in range(int(valid_num/Valid_Bath_Size)):
                valid_output1=cnn(valid_x[m*Valid_Bath_Size:(m+1)*Valid_Bath_Size]).cuda()
                if m == 0: valid_output =valid_output1.cuda()
                else :
                    valid_output=torch.cat((valid_output,valid_output1),0).cuda()
            pred_y = torch.max(valid_output, 1)[1].cuda().data.squeeze()
            accuracy=sum(pred_y==valid_y)/valid_y.size(0)
            accs.append(accuracy)
            recordloss.append(loss.data[0])

            if accuracy > best_validation_accuracy:  #if validation accuracy in this time is better than last time
                best_validation_accuracy = accuracy  # update best accuracy
                last_improvement = total_iterations  # update iteration

                
                torch.save(cnn, 'cnn+augment(best).pkl')          # save model

            print('Epoch:',epoch,'Step:',step,'|train loss:',loss.data[0],'|valid accuracy:',accuracy)
            t+=1
            count.append(t)
        if total_iterations - last_improvement > require_improvement:
            print("no improvment, stopping")
            break_signal=1
            break  
    if break_signal: break


#check the performace of CNN
cnn=torch.load('cnn+augment(best).pkl')
for m in range(int(valid_num/Valid_Bath_Size)):
    valid_output1=cnn(valid_x[m*Valid_Bath_Size:(m+1)*Valid_Bath_Size]).cuda()
    if m == 0: valid_output =valid_output1.cuda()
    else :
        valid_output=torch.cat((valid_output,valid_output1),0).cuda()
pred_y = torch.max(valid_output, 1)[1].cuda().data.squeeze()
accuracy=sum(pred_y==valid_y)/valid_y.size(0)
print('accuracy',accuracy)

np.savetxt("drive1/Colab Notebooks/accs_CNN_sunanda.csv",accs,fmt='%f',delimiter=',')
np.savetxt("drive1/Colab Notebooks/loss_CNN_sunanda.csv",recordloss,fmt='%f',delimiter=',')
np.savetxt("drive1/Colab Notebooks/totalloss_CNN_sunanda.csv",totalloss,fmt='%f',delimiter=',')
