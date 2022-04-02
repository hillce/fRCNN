import time
import copy
import os
import math
import sys
import csv
import argparse
import json
import datetime

import numpy as np
import pandas
import utils
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
parser.add_argument('-dir',nargs='?',metavar='DIR',type=str,default='D:/UKB_Liver/20204_2_0',help='Directory to search for DICOM files')

args = parser.parse_args()

print("Parsed Arguments: ",args.device,args.batch_size,args.start,args.step,args.model)

classDict = {'Bg':0,'Body':1,'Liver':2,'Cyst':3,'Lung':4,'Heart':5,'Spleen':6,'Aorta':7,'Kidney':8,'IVC':9,
                    'Bg ':0,'Body ':1,'Liver ':2,'Cyst ':3,'Lung ':4,'Heart ':5,'Spleen ':6,'Aorta ':7,'Kidney ':8,'IVC ':9}

colorDict = {1:"tab:blue",2:"tab:green",3:"tab:orange",4:"tab:red",5:"tab:purple",6:"tab:brown",7:"tab:pink",8:"tab:olive",9:"tab:cyan"}
textDict = {1:'Body',2:'Liver',3:'Cyst',4:'Lung',5:'Heart',6:'Spleen',7:'Aorta',8:'Kidney',9:'IVC'}

# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone = torchvision.models.mobilenet_v3_large(pretrained=False).features
# backbone = torchvision.models.resnet34(pretrained=True)
# backbone = torchvision.models.densenet121(pretrained=True).features
# backbone = torchvision.models.vgg11_bn(pretrained=True).features

# backbone = torch.nn.Sequential(*(list(backbone.children())[:-2])) # needed for vgg11_bn and resnet34

# backbone.out_channels = 1280 # mobilenet_v2
# backbone.out_channels = 1024 # densenet121
backbone.out_channels = 960 # mobilenet_v3_large
# backbone.out_channels = 512 # vgg11_bn or resnet34

anchorGen = AnchorGenerator(sizes=((32,64,128,256,512),),aspect_ratios=((0.5,1.0,2.0),))

model = FasterRCNN(backbone,num_classes=10,rpn_anchor_generator=anchorGen)#,box_roi_pool=roiPooler)

checkpoint = torch.load('./models/'+args.model+'.pt')
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
del checkpoint

model.eval()
model.to(args.device)

folList = [os.path.join(args.dir,x) for x in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir,x))]
# testList = pandas.read_csv("data_new.csv",header=None)
# eids = testList[0]
# folList = [os.path.join(args.dir,x) for x in eids]

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

dt = datetime.datetime.today()
jsonFile = "Biobank_Bounding_Boxes_{}_{}_{}_{}.json".format(dt.year,dt.month,dt.day,args.model)

with torch.no_grad():
    overall_dict = {}
    for ii,fol in enumerate(folList):
        # try:
        fol_dict = {}
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
        img = np.zeros((img0.shape[0],img0.shape[1],3))
        img[:,:,0] = img0

        ds = pydicom.dcmread(instTime[keys[3]])
        img[:,:,1] = ds.pixel_array

        ds = pydicom.dcmread(instTime[keys[6]])
        img[:,:,2] = ds.pixel_array
        
        img = img[:,:,(2,1,0)].astype(int)
        img = np.transpose(img,(2,0,1))
        inp = torch.tensor(img)
        inp = inp.float()
        inp = inp.to(args.device)
        inp.unsqueeze_(0)
        output = model(inp)

        num = {x:0 for x in textDict.keys()}
        for out in output:
            boxCoord = out['boxes'].cpu()
            boxCoord = boxCoord.numpy()

            labels = out['labels'].cpu()
            labels = labels.numpy()

            scores = out['scores'].cpu()
            scores = scores.numpy()
            for sublabel, subcoords,subScore in zip(labels,boxCoord,scores):
                subcoords = np.append(subcoords, subScore)
                label = "{}_{}".format(textDict[sublabel],num[sublabel])
                num[sublabel] += 1
                fol_dict[label] = list(map(float,subcoords))
        
        overall_dict[fol[-17:]] = fol_dict

        # except:
        #     print("\nError in Folder: {}\n".format(fol[-17:]))
        #     with open("Errors.txt",'a') as f:
        #         f.write("\n{}\n".format(fol[-17:]))

with open(jsonFile,'a') as f:
    json.dump(overall_dict,f)
