# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 20:21:53 2018

@author: Ibtihel, Sunanda, Yimeng
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torchvision import transforms as tfs
from sklearn.model_selection import train_test_split

print(torch.cuda.is_available())
#Hyper Parameters
EPOCH=100
BATCH_SIZE=100
LR=0.001
Valid_Bath_Size=200
TRY=0

best_validation_accuracy = 0.0    # the best validation accuracy in this time
last_improvement = 0             # iteration that has a improvment
require_improvement = 3000    # stop if no improvment in 3000 iteration
total_iterations = 0
break_signal=0
b_x=torch.zeros((BATCH_SIZE,1,28,28))
Modename='ResNet3-19（RES2）.pkl'

###########################################################################################################
#loading data
if TRY :
    train_x=np.zeros((1000,28*28))
    Valid_Bath_Size = 20
else: train_x=np.zeros((50000,28*28))

i=0

#filename1="train_x.csv"  
#plz change the name of the file accordingly; correspond to the ProcessedData/LargestDigit/train_x.csv    
filename1="cs.mcgill.ca/~ksinha4/datasets/kaggle/train_x.csv"                                         #load training data                                   #load training data
with open(filename1,'r',encoding='utf8') as f:
    for line in f.readlines():
        print(i)
        img=np.array(line.split(','))
        img=img.astype(np.float)
       # for k in range(64*64):
       #     if img[k]<240:  img[k]=0
       # img=img/255
        train_x[i]=img
        i+=1
        if TRY:
            if i>=1000: break
    f.close()
# load training label
train_y=np.loadtxt("cs.mcgill.ca/~ksinha4/datasets/kaggle/train_y.csv",dtype=int,delimiter=',')
if TRY:  train_y=train_y[:1000]

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

#####################################################################################
#construct ResNet
# def conv5x5(in_channels,out_channels,stride=1):
#     return nn.Conv2d(in_channels,out_channels,5,1,2)
# Residual Block

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(ResidualBlock,self).__init__()
        self.res1=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,stride,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample

    def forward(self,x):
        residual=x
        out=self.res1(x)
        if self.downsample:
            residual=self.downsample(x)
        out+=residual
        out=self.relu(out)
        return out

# ResNet Module
class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=10):
        super(ResNet, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,16,5,1,4),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.in_channels = 16
        self.layer1=self.make_layer(block,16,blocks=layers[0])              #32*32
        self.layer2=self.make_layer(block,32,blocks=layers[1],stride=2)     #16*16
        self.layer3=self.make_layer(block,64,blocks=layers[2],stride=2)     #8*8
        self.layer4 = self.make_layer(block,128,blocks=layers[3],stride=2)  #4*4
        self.layer5 = self.make_layer(block,256,blocks=layers[4], stride=2) #2*2
        
        self.avg_pool=nn.AvgPool2d(2)
        self.fn=nn.Linear(256,num_classes)
    def make_layer(self,block,out_channnels,blocks,stride=1):
        downsample=None
        if (stride != 1) or (self.in_channels != out_channnels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,out_channnels,3,stride=stride,padding=1),
                nn.BatchNorm2d(out_channnels)
            )
        layers=[]
        layers.append(block(self.in_channels,out_channnels,stride=stride,downsample=downsample))
        self.in_channels=out_channnels
        for i in range(1,blocks):
            layers.append(block(out_channnels,out_channnels))
        return nn.Sequential(*layers)

    def forward(self,x):
        out=self.conv1(x)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        
        out=self.avg_pool(out)
        out=out.view(out.size(0),-1)
        out=self.fn(out)
        return out


# res_layers=[2,3,3,4,5]
res_layers=[2,3,4,6,3]
resnet=ResNet(ResidualBlock,res_layers).cuda()       #GPU
print(resnet)
optimizer=torch.optim.Adam(resnet.parameters(),lr=LR)   #optimize all  parameters
loss_func=nn.CrossEntropyLoss()

# data agumention     randomly choose immages to do following transformation
transform = tfs.Compose([         
    tfs.ToPILImage(),  
    tfs.RandomRotation(180,expand=True),       #rotation
    tfs.RandomResizedCrop(28,scale=(0.8,1)),   #resize to 64*64
    tfs.RandomHorizontalFlip(),                #flip horizontal        
    tfs.ToTensor()]
)
accs=[]
recordloss=[]
totalloss=[]
count=[]
t=0
#training and testing
for epoch in range(EPOCH):
    train_loader = Data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
    for step,(x,y) in enumerate(train_loader):
        for i in range(BATCH_SIZE):
            b_x[i] = transform(x[i])
        b_x=Variable(x).cuda()     #GPU
        b_y=Variable(y).cuda()      #GPU
        total_iterations+=1         #compute iteration times
        output=resnet(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()   #clear gradients for this training step
        loss.backward()         #backpropagation,compute gradients
        totalloss.append(loss.data[0])
        optimizer.step()        #apply gradients
        if step%50==0:
            for m in range(int(valid_num / Valid_Bath_Size)):
                valid_output1 = resnet(valid_x[m * Valid_Bath_Size:(m + 1) * Valid_Bath_Size]).cuda()
                if m == 0:
                    valid_output = valid_output1.cuda()
                else:
                    valid_output = torch.cat((valid_output, valid_output1), 0).cuda()
            pred_y = torch.max(valid_output, 1)[1].cuda().data.squeeze()
            accuracy=sum(pred_y==valid_y)/valid_num
            accs.append(accuracy)
            recordloss.append(loss.data[0])

            if accuracy > best_validation_accuracy:  #if validation accuracy in this time is better than last time
                best_validation_accuracy = accuracy  # update best accuracy
                last_improvement = total_iterations  # update iteration
                
                torch.save(resnet, Modename)          # save model
            print('Epoch:',epoch,'Step:',step,'|train loss:',loss.data[0],'|valid accuracy:',accuracy)
            t+=1
            count.append(t)
        
        if total_iterations - last_improvement > require_improvement:
            print("no improvment, stop training")
            break_signal=1
            break  
 #------- --------------Decaying Learning Rate ----------------------------------#
    if break_signal: break
    if (epoch + 1) % 10 == 0:
        LR /= 3
        optimizer = torch.optim.Adam(resnet.parameters(), lr=LR)

###############################################################################################################
#check the performace of ResNet
resnet=torch.load(Modename)
for m in range(int(valid_num/Valid_Bath_Size)):
    valid_output1=resnet(valid_x[m*Valid_Bath_Size:(m+1)*Valid_Bath_Size]).cuda()
    if m == 0: valid_output =valid_output1.cuda()
    else :
        valid_output=torch.cat((valid_output,valid_output1),0).cuda()
pred_y = torch.max(valid_output, 1)[1].cuda().data.squeeze()
accuracy=sum(pred_y==valid_y)/valid_num
print('accuracy',accuracy)
np.savetxt("accs_resnet.csv",accs,fmt='%f',delimiter=',')
np.savetxt("loss_resnet.csv",recordloss,fmt='%f',delimiter=',')
np.savetxt("totalloss_resnet.csv",totalloss,fmt='%f',delimiter=',')

#-----------------------plot-------------------------------------------------------#
#accuracy
plt.figure()
plt.plot(accs,'r-',label='Valid-accuracy')
plt.axis([0, t, 0,1])
plt.ylabel("Accuracy")
plt.xlabel("Iterations")
plt.title(" ResNet performance(Accuracy)")
plt.legend()
plt.show()

#loss
plt.figure()
plt.plot(recordloss,'g-',label='loss')
plt.axis([0, t, 0,1])
plt.ylabel("Loss")
plt.xlabel("Iterations")
plt.title(" ResNet performance(Loss)")
plt.legend()
plt.show()


#########################################################################################################
input("Check test or not: ")
i=0
test_x=np.zeros((10000,28*28))
filename2="cs.mcgill.ca/~ksinha4/datasets/kaggle/test_x.csv"
with open(filename2,'r',encoding='utf8') as f2:
    for line in f2.readlines():
        print(i)
        img=np.array(line.split(','))
        img=img.astype(np.float)
        #for k in range(64*64):
         #   if img[k]<240:  img[k]=0
       # img=img/255
        test_x[i]=img
        i+=1
    f2.close()

test_x=test_x.reshape(-1,28,28)
test_x=torch.from_numpy(test_x).type(torch.FloatTensor)     #transform to tensor from
test_x=Variable(torch.unsqueeze(test_x,dim=1),volatile=True).cuda()

resnet=torch.load(Modename)
test_num=test_x.shape[0]
Test_Bath_Size=500
for m in range(int(test_num / Test_Bath_Size)):
    valid_output1 = resnet(test_x[m * Test_Bath_Size:(m + 1) * Test_Bath_Size])
    if m == 0:
        valid_output = valid_output1
    else:
        valid_output = torch.cat((valid_output, valid_output1), 0)
pred_y = torch.max(valid_output, 1)[1].cuda().data.squeeze()
pred_y=pred_y.cpu()
pred_y=pred_y.numpy()
np.savetxt("test_y3-19.csv",pred_y,fmt='%d',delimiter=',')





