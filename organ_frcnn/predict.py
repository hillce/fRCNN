import argparse
import typing as t
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader

from DicomDataset import DicomDataset, Rescale, ToTensor, Normalise, collate_var_rois
from config import classes, colorDict, textDict

def predict_args():
    """

    Argparser for test function for fRCNN

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        help="Path to data directory",
        type=str,
        required=True
    )
    parser.add_argument(
        "--state_dict_path",
        help="Path to the model to load",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_directory",
        help="Path to directory to save images to.",
        type=str,
        required=True
    )
    parser.add_argument(
        "--threshold",
        help="Threshold to use for success event of bounding box. Defaults to 0.5",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--device",
        help="Device to run prediction on. Defaults to None, (auto selection)",
        default=None
    )
    parser.add_argument(
        "--predict_csv",
        help="csv file containing the study_ids (sub-folders in data_path) which predictions are needed for",
        type=str,
        default="predict.csv"
    )
    parser.add_argument(
        "--model_backbone",
        help="The model backbone to use for the fRCNN",
        choices=['mobilenet_v2', 'mobilenet_v3', 'resnet34', 'densenet121', 'vgg11'],
        type=str,
        default="mobilenet_v3"
    )
    args = parser.parse_args()

    predict(
        data_path=args.data_path,
        state_dict_path=args.state_dict_path,
        output_directory=args.output_directory,
        threshold=args.threshold,
        device=args.device,
        predict_csv=args.predict_csv,
        model_backbone=args.model_backbone
    )

def predict(
    data_path: str,
    state_dict_path: str, # './models/extra_classes.pt'
    output_directory: str,
    threshold: float = 0.5,
    device: t.Union[t.Type[None], str] = None,
    predict_csv: str = "predict.csv",
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
            ToTensor(),
            Normalise()
        ]
    )

    testDataset = DicomDataset(predict_csv,data_path,transform=test_transforms)
    testLoader = DataLoader(testDataset,batch_size=1,shuffle=True,collate_fn=collate_var_rois)


    #################################################
    ###### Setup Model and Anchors ##################
    #################################################

    print('\33[1;37;40m#'*50)
    print(' Setting up Model and Anchor Generator')
    print('#'*50)

    # Setup the model and pretrain

    if model_backbone == 'mobilenet_v2':
        backbone = torchvision.models.mobilenet_v2().features
        backbone.out_channels = 1280

    elif model_backbone == 'mobilenet_v3':
        backbone = torchvision.models.mobilenet_v3_large().features
        backbone.out_channels = 960

    elif model_backbone == 'resnet34':
        backbone = torchvision.models.resnet34()
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-2])) # needed for vgg11_bn and resnet34
        backbone.out_channels = 512

    elif model_backbone == 'densenet121':
        backbone = torchvision.models.densenet121().features
        backbone.out_channels = 1024

    elif model_backbone == 'vgg11':
        backbone = torchvision.models.vgg11_bn().features
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

    checkpoint = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint

    model.eval()
    model.to(device)


    with torch.no_grad():
        for i, data in enumerate(testLoader):
            print(f"Predicting for {i}: ")
            newImgs = []
            for imgTS in data['image']:
                imgTS = imgTS.float()
                newImgs.append(imgTS.to(device))
            output = model(newImgs)

            for pred,img in zip(output, data['image']):

                boxCoord = pred['boxes'].cpu().detach().numpy()
                label = pred['labels'].cpu().detach().numpy()
                scores = pred['scores'].cpu().detach().numpy()

                _, ax = plt.subplots(1)
                img = img.numpy()
                img = np.transpose(img,(1,2,0))
                ax.imshow(img[:,:,1],vmax=img[:,:,1].max()/2, cmap="gray")

                for lb,coord,sc in zip(label,boxCoord,scores):
                    if sc > threshold:
                        plt.text(coord[0]-2,coord[1]-2,textDict[lb],color=colorDict[lb])
                        rect = patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],linewidth=1,edgecolor=colorDict[lb],facecolor='none')
                        ax.add_patch(rect)
                
                ax.axis("off")
                plt.savefig(os.path.join(output_directory,f"{i}.png"))
                plt.close("all")

            # print(f"\t {boxCoord}, {label}, {scores}")

if __name__ == "__main__":
    predict_args()