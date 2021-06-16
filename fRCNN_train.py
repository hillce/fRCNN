import time
import copy
import os
import utils
import math
import sys

import numpy as np

import torch
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from DicomDataset import DicomDataset, Rescale, ToTensor, RandomCrop
from fRCNN_func import collate_var_rois, is_dist_avail_and_initialized, get_world_size, reduce_dict

def progressBar(batchSize,curIdx,totSetSize,lossVal,tEpoch,t0,stepTotal=10):
    idxMax = totSetSize/batchSize
    chunks = np.arange(0,idxMax,idxMax/(stepTotal))
    #print(chunks)
    #print(curIdx)
    for ii,val in enumerate(chunks):
        if curIdx < val:
            progVal = ii
            break
        if curIdx >= chunks[-1]:
            progVal = len(chunks)
    sys.stdout.write("\r" + "\33[1;37;40m Progress: [{}>{}] {}/{}, Current loss = \33[1;33;40m{:.8f}\33[1;37;40m, Epoch Time: \33[1;33;40m{:.3f} s\33[1;37;40m, Total Time Elapsed: \33[1;33;40m{}\33[1;37;40m".format((progVal-1)*'=',(stepTotal-progVal)*'.',curIdx*batchSize,totSetSize-1,lossVal,time.time()-tEpoch,time.strftime("%H:%M:%S",time.gmtime(time.time()-t0))))

os.system('cls')

print("#"*50)
print(" Parameter Setup")
print("#"*50)


print("\n\33[1;33;40m Running File:\33[1;36;40m {}".format(sys.argv[0]))

try:
    modelSaveName = sys.argv[1]
    print("\33[1;33;40m Saving model to:\33[1;36;40m  ./models/{}.pt".format(modelSaveName))
except IndexError as e:
    print("\033[1;31;40m Index Error found, no model save input found. Please pass model save name when running model. {} \n saving model as Debug".format(e))
    modelSaveName = "Debug"
try:
    dataLocation = sys.argv[2]
    print("\33[1;33;40m Data Location provided. Using:\33[1;36;40m {}".format(dataLocation))
except IndexError as e:
    print("\33[1;33;40m No data Location provided. Using:\33[1;36;40m 'D:/UKB_Liver/20204_2_0'")
    dataLocation = 'D:/UKB_Liver/20204_2_0'

try:
    batch_size = int(sys.argv[3])
    print("\33[1;33;40m Batch Size = \33[1;36;40m {}".format(batch_size))
except IndexError as e:
    batch_size = 2
    print("\33[1;33;40m Batch Size = \33[1;36;40m {}".format(batch_size))

writer = SummaryWriter('runs/{}'.format(modelSaveName)) # Setup Writer for Tensorboard assesment.

print("\33[1;33;40m Backends Enabled?:\33[1;32;40m {}".format(torch.backends.cudnn.enabled))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Setup Device Choice
print("\33[1;33;40m Run on device:\33[1;32;40m {}\n".format(device))


####################################################
############### Data Loading Code ##################
####################################################

print('\33[1;37;40m#'*50)
print(' Loading Data and setting up data generators')
print('#'*50)

trainDataset = None
valDataset = None

trainLoader = None
valLoader = None

validation = False

csvLoc = "./"
trainDir = os.path.join(csvLoc,'train_new.csv')
valDir = os.path.join(csvLoc,'val_new.csv')

trainDataset = DicomDataset(trainDir,dataLocation,transform=transforms.Compose([Rescale(288),ToTensor()]))
valDataset = DicomDataset(valDir,dataLocation,transform=transforms.Compose([Rescale(288),ToTensor()])) # pandas.errors.ParserError: Error tokenizing data. C error: Expected 11 fields in line 7, saw 21

trainLoader = DataLoader(trainDataset,batch_size=batch_size,shuffle=True,collate_fn=collate_var_rois)
valLoader = DataLoader(valDataset,batch_size=batch_size,shuffle=False,collate_fn=collate_var_rois)

if trainLoader:
    print("\n\33[1;32;40m Setup Train Loader. Dataset Size = {}".format(trainDataset.__len__()))
else:
    print("\n\33[1;31;40m No Train Loader. Exiting.....")
    sys.exit()


if valLoader:
    print("\33[1;32;40m Setup Validation Loader. Dataset Size = {}\n".format(valDataset.__len__()))
    validation = True
else:
    print("\33[1;31;40m No Validation Loader!\n")


#################################################
###### Write Examples to Tensorboard ############
#################################################

dataiter = iter(trainLoader)
sample = dataiter.next()
imgGrid = torchvision.utils.make_grid(sample['image'])
writer.add_image('Four Abdominal Scans',imgGrid)

#################################################
###### Setup Model and Anchors ##################
#################################################

print('\33[1;37;40m#'*50)
print(' Setting up Model and Anchor Generator')
print('#'*50)

#classes = ('Body','Liver','Cyst','Lung','Heart','Bg')
#classesDict = {0.:'Body',1.:'Liver',2.:'Cyst',3.:'Lung',4.:'Heart',5.:'Bg'}

classes = ['Bg','Body','Liver','Cyst','Lung','Heart','Spleen','Aorta','Kidney','IVC']
classesDict = {0.:'Bg',1.:'Body',2.:'Liver',3.:'Cyst',4.:'Lung',5.:'Heart',6.:'Spleen',7.:'Aorta',8.:'Kidney',9.:'IVC'}

# classes = ('Bg','Body','Liver','Cyst','Lung','Heart')
# classesDict = {0.:'Bg',1.:'Body',2.:'Liver',3.:'Cyst',4.:'Lung',5.:'Heart'}

# Setup the model and pretrain

# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone = torchvision.models.resnet101(pretrained=True).features
backbone.out_channels = 1280

anchorGen = AnchorGenerator(sizes=((32,64,128,256,512),),aspect_ratios=((0.5,1.0,2.0),))
# roiPooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],output_size=7,sampling_ratio=2)

model = FasterRCNN(backbone,num_classes=len(classes),rpn_anchor_generator=anchorGen)#,box_roi_pool=roiPooler)

model.to(device)

#targetDictList = []
#for ts in sample['labels']:
#    newTargets = ts.numpy()
#    targetDict = {}
#    newBoxes = np.zeros((newTargets.shape[0],4))
#    newLabels = np.zeros(newTargets.shape[0])
#    for ii,cl in enumerate(newTargets):
#        newBoxes[ii,:] = cl[:4]
#        newLabels[ii] = cl[4]
#    targetDict['boxes'] = torch.tensor(newBoxes,dtype=torch.float32,device=device)
#    targetDict['labels'] = torch.tensor(newLabels,dtype=torch.int64,device=device)
#    targetDictList.append(targetDict)
#
#tempImgs = []
#for imgTS in sample['image']:
#    imgTS = imgTS.float()
#    tempImgs.append(imgTS.to(device))

#writer.add_graph(model,(tempImgs,targetDictList))

print("\n\33[1;33;40m",model,"\n")
#writer.close()


#################################################
################## Train Model ##################
#################################################

print('\33[1;37;40m#'*50)
print(' Begin Training')
print('#'*50)


t0 = time.time()
numEpochs = 1000
optimizer = optim.Adam(model.parameters(),lr=0.001)
bestLoss = 1000000000.0

for epoch in range(numEpochs):
    stepTotal = 40
    tEpoch = time.time()
    print("\n\33[1;33;40m Epoch {}:\n".format(epoch+1))
    lossVal = 10000.0
    print("\33[1;36;40m #### Training Phase ####")
    model.train(True)
    avgLoss = 0
    for i, data in enumerate(trainLoader):
        #optimizer.zero_grad()
        
        targetDictList = []
        for ts in data['labels']:
            newTargets = ts.numpy()
            targetDict = {}
            newBoxes = np.zeros((newTargets.shape[0],4))
            newLabels = np.zeros(newTargets.shape[0])
            for ii,cl in enumerate(newTargets):
                newBoxes[ii,:] = cl[:4]
                newLabels[ii] = cl[4]
            targetDict['boxes'] = torch.tensor(newBoxes,dtype=torch.float32,device=device)
            targetDict['labels'] = torch.tensor(newLabels,dtype=torch.int64,device=device)
            targetDictList.append(targetDict)

        newImgs = []
        for imgTS in data['image']:
            imgTS = imgTS.float()
            newImgs.append(imgTS.to(device))

        lossDict = model(newImgs,targetDictList)
        losses = sum(loss for loss in lossDict.values())

        lossDictRed = reduce_dict(lossDict)
        lossesRed = sum(loss for loss in lossDictRed.values())

        lossVal = lossesRed.item()
        avgLoss += lossVal

        if not math.isfinite(lossVal):
            print(" Loss is {}, stopping training".format(lossVal))
            print(lossDictRed)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if i % (trainDataset.__len__()//stepTotal) == 1:
            progressBar(batch_size,i,trainDataset.__len__(),lossVal,tEpoch,t0,stepTotal=stepTotal)
    
    avgLoss /= trainDataset.__len__()
    writer.add_scalar('Loss/train',avgLoss,epoch)
    writer.add_scalars('LossTrain',lossDict,epoch)

    if validation:
        stepTotal = 10
        print("\n\33[1;36;40m #### Validation Phase ####")
        avgLoss = 0
        with torch.no_grad():
            for i, data in enumerate(valLoader):
                targetDictList = []
                for ts in data['labels']:
                    newTargets = ts.numpy()
                    targetDict = {}
                    newBoxes = np.zeros((newTargets.shape[0],4))
                    newLabels = np.zeros(newTargets.shape[0])
                    for ii,cl in enumerate(newTargets):
                        newBoxes[ii,:] = cl[:4]
                        newLabels[ii] = cl[4]
                    targetDict['boxes'] = torch.tensor(newBoxes,dtype=torch.float32,device=device)
                    targetDict['labels'] = torch.tensor(newLabels,dtype=torch.int64,device=device)
                    targetDictList.append(targetDict)

                newImgs = []
                for imgTS in data['image']:
                    imgTS = imgTS.float()
                    newImgs.append(imgTS.to(device))

                lossDict = model(newImgs,targetDictList)

                #print(lossDict)
                losses = sum(loss for loss in lossDict.values())

                lossDictRed = reduce_dict(lossDict)
                lossesRed = sum(loss for loss in lossDictRed.values())

                lossVal = lossesRed.item()
                avgLoss += lossVal
                if i % (valDataset.__len__()//stepTotal) == 1:
                    progressBar(batch_size,i,valDataset.__len__(),lossVal,tEpoch,t0,stepTotal=stepTotal)

            avgLoss /= valDataset.__len__()
            writer.add_scalar('Loss/validation',avgLoss,epoch)
            writer.add_scalars('LossVal',lossDict,epoch)

    if avgLoss < bestLoss:
        print("\n\33[1;32;40m Loss is \33[1;33;40m{:.8f}\33[1;32;40m, Saving Model.".format(avgLoss))
        bestLoss = avgLoss
        torch.save({'epoch':epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss':bestLoss
                    },"./models/{}.pt".format(modelSaveName))

    else:
        print("\n\33[1;31;40m Val loss is higher, \33[1;33;40m{:.8f} > {:.8f}".format(avgLoss,bestLoss))

print('Finished Training')
