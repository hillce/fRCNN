import argparse
from operator import mod
from typing import Type, Union
import time
import math
import sys

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.init import xavier_uniform_, zeros_
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from organ_frcnn.DicomDataset import DicomDataset, Rescale, ToTensor, Normalise, RandomCrop
from organ_frcnn.fRCNN_func import collate_var_rois, reduce_dict
from organ_frcnn.config import classes
from organ_frcnn.utils import progressBar


def train_args():
    """
    
    Argparser for train function for fRCNN

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="Name to save the model to.",
        type=str,
        required=True
    )
    parser.add_argument(
        "--data_path",
        help="Data directory path",
        type=str,
        default="D:/UKB_Liver/20204_2_0"
    )
    parser.add_argument(
        "--pretrained",
        help="Don't use imagenet pretrained backbone.",
        action="store_false",
        default=True
    )
    parser.add_argument(
        "--lr",
        help="Learning rate for network",
        type=float,
        default=1e-3
    )
    parser.add_argument(
        "--b_size",
        help="Batch size for training.",
        type=int,
        default=4,
        dest="batch_size"
    )
    parser.add_argument(
        "-nE",
        "--num_epochs",
        help="Number of epochs to train model for.",
        default=100,
        dest="num_epochs"
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Device to use, assumes a single device available",
        type=str,
        default="cuda:0",
        dest="device"
    )
    parser.add_argument(
        "--train_csv",
        help="path to train csv",
        type=str,
        default="train.csv"
    )
    parser.add_argument(
        "--val_csv",
        help="path to val csv",
        type=str,
        default="val.csv"
    )
    parser.add_argument(
        "--model_backbone",
        help="The model backbone to use from list ['mobilenet_v2', 'mobilenet_v3', 'resnet34', 'densenet121', 'vgg11']",
        choices=['mobilenet_v2', 'mobilenet_v3', 'resnet34', 'densenet121', 'vgg11'],
        default='mobilenet_v3',
    )
    parser.add_argument(
        "--random_crop",
        help="Perform a random crop on the data during training",
        action='store_true'
    )

    args = parser.parse_args()

    train(
        model_save_name=args.model_name,
        data_path=args.data_path,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        model_backbone=args.model_backbone,
        num_epochs=args.num_epochs,
        random_crop=args.random_crop
    )


def train(
    model_save_name: str,
    data_path: str,
    train_csv: str,
    val_csv: str,
    pretrained: bool = True,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: Union[Type[None], str] = None,
    model_backbone: str = 'mobilenet_v3',
    num_epochs: int = 100,
    step_total_train: int = 40,
    step_total_val: int = 10,
    random_crop: bool = False
    ):
    """Train function for fRCNN

    Args:
        model_save_name (str): model save name.
        data_path (str): path to data folder.
        train_csv (str): path to train csv.
        val_csv (str): path to val csv.
        pretrained (bool, optional): Whether to pretrain backbones on image net.
        batch_size (int, optional): batch size of training and validation. Defaults to 4.
        lr (float, optional): Initial learning rate. Defaults to 1e-3
        device (Union[Type[None], str], optional): Device to train on, will auto select if blank. Defaults to None.
        model_backbone (str, optional): model backbone to use. Selection from: ['mobilenet_v2', 'mobilenet_v3', 'resnet34', 'densenet121', 'vgg11']. Defaults to 'mobilenet_v3'.
        num_epochs (int, optional): Number of epochs to train for. Defaults to 100.
        step_total_train (int, optional): Number of steps in train progress bar. Defaults to 40.
        step_total_val (int, optional): Number of steps in val progress bar. Defaults to 10.
        random_crop (bool, optional): Whether to perform a random crop on data. Defaults to False.
    """

    assert model_backbone in ['mobilenet_v2', 'mobilenet_v3', 'resnet34', 'densenet121', 'vgg11'], \
        "model_backbone invalid, please choose from ['mobilenet_v2', 'mobilenet_v3', 'resnet34', 'densenet121', 'vgg11']"

    print("#"*50)
    print(" Parameter Setup")
    print("#"*50)

    writer = SummaryWriter('runs/{}'.format(model_save_name)) # Setup Writer for Tensorboard assesment.

    print("\33[1;33;40m Backends Enabled?:\33[1;32;40m {}".format(torch.backends.cudnn.enabled))

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

    if random_crop:
        transforms_train = transforms.Compose(
            [
                Rescale(288),
                RandomCrop(256),
                Normalise(),
                ToTensor()
            ]
        )
    else:
        transforms_train = transforms.Compose(
            [
                Rescale(288),
                Normalise(),
                ToTensor()
            ]
        )

    transforms_val = transforms.Compose(
        [
            Rescale(288),
            Normalise(),
            ToTensor()
        ]
    )

    trainDataset = DicomDataset(train_csv,data_path,transform=transforms_train)
    valDataset = DicomDataset(val_csv,data_path,transform=transforms_val)

    trainLoader = DataLoader(trainDataset,batch_size=batch_size,shuffle=True,collate_fn=collate_var_rois)
    valLoader = DataLoader(valDataset,batch_size=batch_size,shuffle=False,collate_fn=collate_var_rois)
    print("\n\33[1;32;40m Setup Train Loader. Dataset Size = {}".format(trainDataset.__len__()))
    print("\33[1;32;40m Setup Validation Loader. Dataset Size = {}\n".format(valDataset.__len__()))

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

    # Setup the model and pretrain

    if model_backbone == 'mobilenet_v2':
        backbone = torchvision.models.mobilenet_v2(pretrained=pretrained).features
        backbone.out_channels = 1280

    elif model_backbone == 'mobilenet_v3':
        backbone = torchvision.models.mobilenet_v3_large(pretrained=pretrained).features
        backbone.out_channels = 960

    elif model_backbone == 'resnet34':
        backbone = torchvision.models.resnet34(pretrained=pretrained)
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-2])) # needed for vgg11_bn and resnet34
        backbone.out_channels = 512

    elif model_backbone == 'densenet121':
        backbone = torchvision.models.densenet121(pretrained=pretrained).features
        backbone.out_channels = 1024

    elif model_backbone == 'vgg11':
        backbone = torchvision.models.vgg11_bn(pretrained=pretrained).features
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

    model.to(device)

    if not pretrained:
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                xavier_uniform_(m.weight.data)
                print(m)
                if m.bias is not None:
                    zeros_(m.bias.data)
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight.data)
                if m.bias.data is not None:
                    zeros_(m.bias.data)   

        model.apply(weights_init)

    print("\n\33[1;33;40m",model,"\n")

    #################################################
    ################## Train Model ##################
    #################################################

    print('\33[1;37;40m#'*50)
    print(' Begin Training')
    print('#'*50)

    t0 = time.time()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    bestLoss = np.inf

    for epoch in range(num_epochs):
        tEpoch = time.time()

        print("\n\33[1;33;40m Epoch {}:\n".format(epoch+1))
        print("\33[1;36;40m #### Training Phase ####")

        model.train()
        avgLoss = 0

        for i, data in enumerate(trainLoader):
            
            targetDictList = [] #REFACTOR: Data formatting into DicomDataset
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
            if i % (trainDataset.__len__()//step_total_train) == 1:
                progressBar(batch_size,i,trainDataset.__len__(),lossVal,tEpoch,t0,stepTotal=step_total_train)
        
        avgLoss /= trainDataset.__len__()
        writer.add_scalar('Loss/train',avgLoss,epoch)
        writer.add_scalars('LossTrain',lossDict,epoch)

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

                losses = sum(loss for loss in lossDict.values())

                lossDictRed = reduce_dict(lossDict)
                lossesRed = sum(loss for loss in lossDictRed.values())

                lossVal = lossesRed.item()
                avgLoss += lossVal
                if i % (valDataset.__len__()//step_total_val) == 1:
                    progressBar(batch_size,i,valDataset.__len__(),lossVal,tEpoch,t0,stepTotal=step_total_val)

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
                        },"./models/{}.pt".format(model_save_name))

        else:
            print("\n\33[1;31;40m Val loss is higher, \33[1;33;40m{:.8f} > {:.8f}".format(avgLoss,bestLoss))
            torch.save({'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':bestLoss
                        },"./models/{}_latest.pt".format(model_save_name))

    print('Finished Training')


if __name__ == "__main__":
    train_args()