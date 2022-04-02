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

batch_size = 4
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

model = FasterRCNN(backbone,num_classes=10,rpn_anchor_generator=anchorGen)#,box_roi_pool=roiPooler)

checkpoint = torch.load('./models/extra_classes.pt')
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
del checkpoint

classDict = {'Bg':0,'Body':1,'Liver':2,'Cyst':3,'Lung':4,'Heart':5,'Spleen':6,'Aorta':7,'Kidney':8,'IVC':9,
                    'Bg ':0,'Body ':1,'Liver ':2,'Cyst ':3,'Lung ':4,'Heart ':5,'Spleen ':6,'Aorta ':7,'Kidney ':8,'IVC ':9}

colorDict = {1:"tab:blue",2:"tab:green",3:"tab:orange",4:"tab:red",5:"tab:purple",6:"tab:brown",7:"tab:pink",8:"tab:olive",9:"tab:cyan"}
textDict = {1:'Body',2:'Liver',3:'Cyst',4:'Lung',5:'Heart',6:'Spleen',7:'Aorta',8:'Kidney',9:'IVC'}

model.eval()
model.to(device)
with torch.no_grad():
    dataiter = iter(testLoader)
    sample = dataiter.next()
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
    label = pred['labels'].cpu().detach().numpy()
    scores = pred['scores'].cpu().detach().numpy()

    print(pred)

    fig, ax = plt.subplots(1)
    img = img.numpy()
    img = np.transpose(img,(1,2,0))
    #ax.imshow(np.transpose(img,axes=[1,2,0]))
    ax.imshow(img[:,:,:],vmax=img[:,:,:].max()/2)
    for lb,coord,sc in zip(label,boxCoord,scores):
        if sc > 0.5:
            plt.text(coord[0]-2,coord[1]-2,textDict[lb],color=colorDict[lb])
            rect = patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],linewidth=1,edgecolor=colorDict[lb],facecolor='none')
            ax.add_patch(rect)
    plt.show()