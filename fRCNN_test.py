import time
import copy
import os
import utils
import math
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

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
from fRCNN_func import collate_var_rois

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Setup Device Choice
#device = torch.device("cpu")
print(device)

batch_size = 2
csvLoc = "./"
testDir = os.path.join(csvLoc,'test.csv')

testDataset = DicomDataset(testDir,'D:/UKB_Liver/20204_2_0',transform=transforms.Compose([Rescale(256),ToTensor()]))
testLoader = DataLoader(testDataset,batch_size=batch_size,shuffle=True,collate_fn=collate_var_rois)

#classes = ('Body','Liver','Cyst','Lung','Heart','Bg')
#classesDict = {0.:'Body',1.:'Liver',2.:'Cyst',3.:'Lung',4.:'Heart',5.:'Bg'}

classes = ('Body','Liver','Cyst','Lung','Heart')
classesDict = {0.:'Body',1.:'Liver',2.:'Cyst',3.:'Lung',4.:'Heart'}

# Setup the model and pretrain
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280

anchorGen = AnchorGenerator(sizes=((32,64,128,256,512),),aspect_ratios=((0.5,1.0,2.0),))
# roiPooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],output_size=7,sampling_ratio=2)

model = FasterRCNN(backbone,num_classes=5,rpn_anchor_generator=anchorGen)#,box_roi_pool=roiPooler)
optimizer = optim.Adam(model.parameters(),lr=0.001)

checkpoint = torch.load('./models/fRCNN.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
model.to(device)
dataiter = iter(testLoader)
sample = dataiter.next()
print(sample)
newImgs = []
for imgTS in sample['image']:
    imgTS = imgTS.float()
    newImgs.append(imgTS.to(device))
output = model(newImgs)

for pred,img in zip(output,sample['image']):

    if device == torch.device("cuda:0"):
        boxCoord = pred['boxes'].cpu()
        boxCoord = boxCoord.detach().numpy()
    else:
        boxCoord = pred['boxes'].detach().numpy()
    label = pred['labels']
    fig, ax = plt.subplots(1)
    img = img.numpy()
    #ax.imshow(np.transpose(img,axes=[1,2,0]))
    ax.imshow(img[0,:,:])
    for coord in boxCoord:
        rect = patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.show()