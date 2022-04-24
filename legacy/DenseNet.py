#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt

class DenseNet(nn.Module):
    def __init__(self, denseBlocks, denseLayers, numClasses, growthRate = 10, dropRate = 0.3, num_anchors = 32, imgDim = (256,256), poolRegions=7, numRois=300):
        super(DenseNet,self).__init__()
        self.denseBlocks = denseBlocks
        self.denseLayers = denseLayers
        self.numClasses = numClasses
        self.gR = growthRate
        self.numChannels = growthRate * 2
        self.numFeatures = self.numChannels
        self.numAnchors = num_anchors
        self.inpChannels = 3
        self.dropRate = dropRate
        self.imgDim = imgDim
        self.poolRegions = poolRegions
        self.numRois = numRois

        self.dp = nn.Dropout(p=self.dropRate)
        # 4 * 4 DenseNet

        # Initial Setup
        self.conv1 = nn.Conv2d(3,self.numChannels,7,padding=3,stride=2)
        self.maxPool1 = nn.MaxPool2d(3,2,padding=1)
        self.avgPool = nn.AvgPool2d(2,stride=2)

        # Block 1
        self.conv2 = nn.Conv2d(self.numChannels,self.numChannels+self.gR,3,padding=1)
        self.bn2 = nn.BatchNorm2d(self.numChannels+self.gR)

        self.conv3 = nn.Conv2d(2*(self.numChannels)+self.gR,self.numChannels+self.gR*2,3,padding=1)
        self.bn3 = nn.BatchNorm2d(self.numChannels+self.gR*2)

        self.conv4 = nn.Conv2d(3*(self.numChannels)+3*self.gR,self.numChannels+self.gR*3,3,padding=1)
        self.bn4 = nn.BatchNorm2d(self.numChannels+self.gR*3)
        
        self.conv5 = nn.Conv2d(4*(self.numChannels)+6*self.gR,self.numChannels+self.gR*3,1)
        self.numChannels += self.gR*3
        
        # Block 2
        self.conv6 = nn.Conv2d(self.numChannels,self.numChannels+self.gR,3,padding=1)
        self.bn6 = nn.BatchNorm2d(self.numChannels+self.gR)

        self.conv7 = nn.Conv2d(2*(self.numChannels)+self.gR,self.numChannels+self.gR*2,3,padding=1)
        self.bn7 = nn.BatchNorm2d(self.numChannels+self.gR*2)

        self.conv8 = nn.Conv2d(3*(self.numChannels)+3*self.gR,self.numChannels+self.gR*3,3,padding=1)
        self.bn8 = nn.BatchNorm2d(self.numChannels+self.gR*3)
        
        self.conv9 = nn.Conv2d(4*(self.numChannels)+6*self.gR,self.numChannels+self.gR*3,1)
        self.numChannels += self.gR*3

        # Block 3
        self.conv10 = nn.Conv2d(self.numChannels,self.numChannels+self.gR,3,padding=1)
        self.bn10 = nn.BatchNorm2d(self.numChannels+self.gR)

        self.conv11 = nn.Conv2d(2*(self.numChannels)+self.gR,self.numChannels+self.gR*2,3,padding=1)
        self.bn11 = nn.BatchNorm2d(self.numChannels+self.gR*2)

        self.conv12 = nn.Conv2d(3*(self.numChannels)+3*self.gR,self.numChannels+self.gR*3,3,padding=1)
        self.bn12 = nn.BatchNorm2d(self.numChannels+self.gR*3)
        
        self.conv13 = nn.Conv2d(4*(self.numChannels)+6*self.gR,self.numChannels+self.gR*3,1)
        self.numChannels += self.gR*3

        # Block 4
        self.conv14 = nn.Conv2d(self.numChannels,self.numChannels+self.gR,3,padding=1)
        self.bn14 = nn.BatchNorm2d(self.numChannels+self.gR)

        self.conv15 = nn.Conv2d(2*(self.numChannels)+self.gR,self.numChannels+self.gR*2,3,padding=1)
        self.bn15 = nn.BatchNorm2d(self.numChannels+self.gR*2)

        self.conv16 = nn.Conv2d(3*(self.numChannels)+3*self.gR,self.numChannels+self.gR*3,3,padding=1)
        self.bn16 = nn.BatchNorm2d(self.numChannels+self.gR*3)
        
        self.conv17 = nn.Conv2d(4*(self.numChannels)+6*self.gR,self.numChannels+self.gR*3,1)
        #self.numChannels += self.gR*3

        # Classification Layer
        self.flatten = nn.Flatten()
        fc1_in = (4*(self.numChannels)+6*self.gR)*(imgDim[0]//(2**(self.denseBlocks+2-1)))*(imgDim[1]//(2**(self.denseBlocks+2-1)))
        self.fc1 = nn.Linear(fc1_in,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc_class = nn.Linear(4096,self.numClasses)
        self.fc_regr = nn.Linear(4096,4*(self.numClasses-1))

        # RPN Layer
        self.convRPN_1 = nn.Conv2d(4*(self.numChannels)+6*self.gR,256,3,padding=1)
        self.regrRPN = nn.Conv2d(256,self.numAnchors*4,(1,1))
        self.classRPN = nn.Conv2d(256,self.numAnchors,(1,1))

    def forward(self,x,network):
        # Initial Setup
        print("Initial Size",x.size())
        x = self.conv1(x)
        print("After Conv 1",x.size())
        x1 = self.maxPool1(x)
        print("Initial pre-Blocks Size",x1.size())

        # Block 1
        x2 = F.relu(self.bn2(self.dp(self.conv2(x1))))
        x = torch.cat((x1,x2),1)
        x3 = F.relu(self.bn3(self.dp(self.conv3(x))))
        x = torch.cat((x1,x2,x3),1)
        x4 = F.relu(self.bn4(self.dp(self.conv4(x))))
        x = torch.cat((x1,x2,x3,x4),1)
        
        # Trans 1
        x5 = self.dp(self.conv5(x))
        x1 = self.avgPool(x5)
        print("Final Block 1 Size:",x1.size())

        # Block 2
        x2 = F.relu(self.bn6(self.dp(self.conv6(x1))))
        x = torch.cat((x1,x2),1)
        x3 = F.relu(self.bn7(self.dp(self.conv7(x))))
        x = torch.cat((x1,x2,x3),1)
        x4 = F.relu(self.bn8(self.dp(self.conv8(x))))
        x = torch.cat((x1,x2,x3,x4),1)
        
        # Trans 2
        x5 = self.dp(self.conv9(x))
        x1 = self.avgPool(x5)
        print("Final Block 2 Size:",x1.size())

        # Block 3
        x2 = F.relu(self.bn10(self.dp(self.conv10(x1))))
        x = torch.cat((x1,x2),1)
        x3 = F.relu(self.bn11(self.dp(self.conv11(x))))
        x = torch.cat((x1,x2,x3),1)
        x4 = F.relu(self.bn12(self.dp(self.conv12(x))))
        x = torch.cat((x1,x2,x3,x4),1)
        
        # Trans 3
        x5 = self.dp(self.conv13(x))
        x1 = self.avgPool(x5)
        print("Final Block 3 Size:",x1.size())

        # Block 4
        x2 = F.relu(self.bn14(self.dp(self.conv14(x1))))
        x = torch.cat((x1,x2),1)
        x3 = F.relu(self.bn15(self.dp(self.conv15(x))))
        x = torch.cat((x1,x2,x3),1)
        x4 = F.relu(self.bn16(self.dp(self.conv16(x))))
        x = torch.cat((x1,x2,x3,x4),1)
        print("Final Block 4 Size:",x.size())

        # Classification Layer
        if network == "classifier":
            x = self.classifier(x)
        elif network == "rpn":
            x = self.rpn(x)
        elif network == "all":
            x = self.all(x)
        
        return x

    def rpn(self,x):
        x = F.relu(self.convRPN_1(x))
        x_cl = F.sigmoid(self.classRPN(x))
        x_regr = self.regrRPN(x)

        return [x_cl,x_regr]

    def classifier(self,x):
        rpc = RoiPoolingConv(self.poolRegions,self.numRois)
        x = rpc.call(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x_cl = F.softmax(self.fc_class(x))
        x_regr = self.fc_regr(x)

        return [x_cl,x_regr]

    def all(self,x):
        x = F.relu(self.convRPN_1(x))
        rpc = RoiPoolingConv(self.poolRegions,self.numRois)
        x = rpc.call(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x_cl = F.softmax(self.fc_class(x))
        x_regr = self.fc_regr(x)

        return [x_cl,x_regr]

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions but the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class RoiPoolingConv(nn.Module):
    def __init__(self,poolSize,numRois,numChannels=3):
        super(RoiPoolingConv,self).__init__()
        self.poolSize = poolSize
        self.numRois = numRois

        self.numChannels = numChannels

    def call(self, x, mask=None):
        print(x.size())
        assert(len(x) == 2) # x from forward in Net, Input_Rois from outside
        outputs = []
        for roiIdx in range(self.numRois):
            xCoord = x[1][0,roiIdx,0]
            yCoord = x[1][0,roiIdx,1]
            width = x[1][0,roiIdx,2]
            height = x[1][0,roiIdx,3]

            rowLen = width/float(self.poolSize)
            colLen = height/float(self.poolSize)

            xCoord = xCoord.type(torch.int32)
            yCoord = yCoord.type(torch.int32)
            width = width.type(torch.int32)
            height= height.type(torch.int32)         

            rs = tf.image.resize_images(x[0][:,yCoord:yCoord+height,xCoord:xCoord+width],(self.poolSize,self.poolSize))
            outputs.append(rs)

        finalOutput = torch.cat(outputs,dim=0)
        finalOutput = torch.reshape(finalOutput,(1,self.numRois,self.poolSize,self.poolSize,self.numChannels))

        return finalOutput


net = DenseNet(4,3,4,network="rpn")
input = torch.randn(10, 3, 256, 256)
out = net(input)
net.forward()
print(out[0].size())
print(out[1].size())
out_class = out[1]
fig, axes = plt.subplots(nrows=2,ncols=2)
for i in range(out_class.size()[1]):
    if i < 4:
        out_np = out_class[0,i,:,:].detach().numpy()
        axes[i//2,i%2].imshow(out_np)
    else:
        break

plt.show()
