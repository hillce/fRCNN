import argparse
import time
import copy
import typing as t
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
from organ_frcnn.DicomDataset import DicomDataset, Rescale, ToTensor, Normalise, collate_var_rois
from organ_frcnn.config import classes, classDict, colorDict, textDict

def test_args():
    """

    Argparser for test function for fRCNN

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="Name of the model to load and test",
        type=str,
        required=True
    )

def test(
    data_path: str,
    state_dict_path: str, # './models/extra_classes.pt'
    batch_size: int = 4,
    device: t.Union[t.Type[None], str] = None,
    test_csv: str = "test.csv",
    model_backbone: str = 'mobilenet_v3',
):
    assert model_backbone in ['mobilenet_v2', 'mobilenet_v3', 'resnet34', 'densenet121', 'vgg11'], \
        "model_backbone invalid, please choose from ['mobilenet_v2', 'mobilenet_v3', 'resnet34', 'densenet121', 'vgg11']"

    print("#"*50)
    print(" Parameter Setup")
    print("#"*50)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Setup Device Choice
    else:
        device = torch.device(device)

    print("\33[1;33;40m Run on device:\33[1;32;40m {}\n".format(device))

    ####################################################
    ############### Data Loading Code ##################
    ####################################################

    print('\33[1;37;40m#'*50)
    print(' Loading Data and setting up data generators')
    print('#'*50)

    test_transforms = transforms.Compose(
        [
            Rescale(288),
            Normalise(),
            ToTensor()
        ]
    )

    testDataset = DicomDataset(test_csv,data_path,transform=test_transforms)
    testLoader = DataLoader(testDataset,batch_size=batch_size,shuffle=True,collate_fn=collate_var_rois)


    #################################################
    ###### Setup Model and Anchors ##################
    #################################################

    print('\33[1;37;40m#'*50)
    print(' Setting up Model and Anchor Generator')
    print('#'*50)

    # Setup the model and pretrain

    if model_backbone == 'mobilenet_v2':
        backbone = torchvision.models.mobilenet_v2(pretrained=False).features
        backbone.out_channels = 1280

    elif model_backbone == 'mobilenet_v3':
        backbone = torchvision.models.mobilenet_v3_large(pretrained=False).features
        backbone.out_channels = 960

    elif model_backbone == 'resnet34':
        backbone = torchvision.models.resnet34(pretrained=False)
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-2])) # needed for vgg11_bn and resnet34
        backbone.out_channels = 512

    elif model_backbone == 'densenet121':
        backbone = torchvision.models.densenet121(pretrained=False).features
        backbone.out_channels = 1024

    elif model_backbone == 'vgg11':
        backbone = torchvision.models.vgg11_bn(pretrained=False).features
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-2])) # needed for vgg11_bn and resnet34
        backbone.out_channels = 512

    anchorGen = AnchorGenerator(sizes=((32,64,128,256,512),),aspect_ratios=((0.5,1.0,2.0),))
    # roiPooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],output_size=7,sampling_ratio=2)

    model = FasterRCNN(
        backbone,
        num_classes=len(classes),
        rpn_anchor_generator=anchorGen,
        #box_roi_pool=roiPooler,
    )

    #################################################
    ###### Load Trained Model      ##################
    #################################################

    print('\33[1;37;40m#'*50)
    print(' Loading Trained Model')
    print('#'*50)

    checkpoint = torch.load(state_dict_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint

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