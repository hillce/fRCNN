import time
import copy
import os
import utils
import math
import sys
import csv
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pydicom

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

parser = argparse.ArgumentParser()
parser.add_argument('-d','--device',nargs='?',metavar='D',type=str,default="cuda:0",choices=["cuda:0","cpu"],help='Select device to use: cuda:0 or cpu')
parser.add_argument('-b','--batch_size',nargs='?',metavar='B',type=int,default=2,help='Choose batch size for the network. Higher batch sizes, use more memory')
parser.add_argument('-s','--start',nargs='?',metavar='S',type=int,default=0,help='Choose start point to compute boxes for csv')
parser.add_argument('--step',nargs='?',metavar='ST',type=int,default=0,help='Choose number of dicoms to process in one go.')
parser.add_argument('-m','--model',metavar='MOD',type=str,default='Random_Validation_2',help='Learnt model to run')
parser.add_argument('-dir',nargs='?',metavar='DIR',type=str,default='C:/Users/shug4421/UKB_Liver/shMOLLI_10765/Data',help='Directory to search for DICOM files')
args = parser.parse_args()

print("Parsed Arguments: ",args.device,args.batch_size,args.start,args.step,args.model)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Setup Device Choice
colorScheme = {0:'r',1:"g",2:'b',3:'m',4:'y',5:'c'}

#labelDict = {0:'Body',1:'Liver',2:'Cyst',3:'Lungs',4:'Heart',5:'Bg'}
#labelDict = {0:'Body',1:'Liver',2:'Cyst',3:'Lungs',4:'Heart'}
labelDict = {0:'Bg',1:'Body',2:'Liver',3:'Cyst',4:'Lungs',5:'Heart'}


# Setup the model and pretrain
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280

anchorGen = AnchorGenerator(sizes=((32,64,128,256,512),),aspect_ratios=((0.5,1.0,2.0),))
roiPooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],output_size=7,sampling_ratio=2)

model = FasterRCNN(backbone,num_classes=6,rpn_anchor_generator=anchorGen,box_roi_pool=roiPooler)
optimizer = optim.Adam(model.parameters(),lr=0.001)

checkpoint = torch.load('./models/'+args.model+'.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
model.to(args.device)
print(model)

sys.exit()

folList = [os.path.join(args.dir,x) for x in os.listdir(args.dir)]
print("Number of shMOLLI acquisitions Found: {}".format(len(folList)))

if args.step:
    folList = folList[args.start:(args.start+args.step)]
    print("Processing Subset: {} -> {}".format(args.start,args.start + args.step))
else:
    folList = folList[args.start:]
    print("Processing Subset: {} -> {}".format(args.start,len(folList)+args.start))

if args.start == 0:
    fileTag = 'w'
else:
    fileTag = 'a'

with open('biobankBoundingBoxes.csv',fileTag,newline='') as f:
    csvObj = csv.DictWriter(f,fieldnames=['File','Organs Found','Scores','Coordinates'])
    if args.start == 0:
        csvObj.writeheader()
    for ii,fol in enumerate(folList):
        sys.stdout.write("\r Completed {}/{}".format(ii,len(folList)))
        dcmList = [os.path.join(fol,x) for x in os.listdir(fol) if x.endswith('.dcm')]
        instTime = {}
        for dicom in dcmList:
            ds = pydicom.dcmread(dicom,stop_before_pixels=True)
            if 'M' in ds.ImageType:
                instTime[float(ds.InstanceCreationTime)] = dicom

        keys = np.sort(list(instTime.keys()),axis=None)
        ds = pydicom.dcmread(instTime[keys[0]])
        img0 = ds.pixel_array
        print(np.shape(img0))
        time.sleep(100)
        img = np.zeros((img0.shape[0],img0.shape[1],3))
        img[:,:,0] = img0

        ds = pydicom.dcmread(instTime[keys[3]])
        img[:,:,1] = ds.pixel_array

        ds = pydicom.dcmread(instTime[keys[6]])
        img[:,:,2] = ds.pixel_array
        
        img = img[:,:,(2,1,0)].astype(np.int)
        img = np.transpose(img,(2,0,1))
        inp = torch.tensor(img)
        inp = inp.float()
        inp = inp.to(args.device)

        output = model([inp])
        for out in output:
            boxCoord = out['boxes'].cpu()
            boxCoord = boxCoord.detach().numpy()

            labels = out['labels'].cpu()
            labels = labels.detach().numpy()

            scores = out['scores'].cpu()
            scores = scores.detach().numpy()

            #fig, ax = plt.subplots(1)
            #ax.imshow(img[0,:,:])
            orgFound = []
            for coord,lab,sc in zip(boxCoord,labels,scores):
                orgFound.append(labelDict[lab])
            #    rect = patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],linewidth=1,edgecolor=colorScheme[lab],facecolor='none')
            #    ax.add_patch(rect)
            #plt.show()

            folStr = fol[47:]
            csvObj.writerow({'File':folStr,'Organs Found':orgFound,'Scores':scores,'Coordinates':boxCoord})
            #fig.savefig('./Batch/'+folStr+'.png')
            #plt.close()