import sys
import argparse
import typing as t

import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader

from DicomDataset import DicomDataset, Rescale, ToTensor, Normalise, collate_var_rois
from config import classes
from metrics import IoU, recall_curve


def test_args():
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
        "--test_csv",
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


    test(
        state_dict_path=args.state_dict_path,
        data_path=args.data_path,
        test_csv=args.test_csv,
        device=args.device,
        model_backbone=args.model_backbone,
        threshold=args.threshold
    )


def test(
    state_dict_path: str,
    data_path: str,
    test_csv: str,
    device: t.Union[t.Type[None], str] = None,
    model_backbone: str = 'mobilenet_v3',
    threshold: float = 0.5
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

    testDataset = DicomDataset(test_csv,data_path,transform=test_transforms)
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

    model = FasterRCNN(
        backbone,
        num_classes=len(classes),
        rpn_anchor_generator=anchorGen,
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

    anchorGen = AnchorGenerator(sizes=((32,64,128,256,512),),aspect_ratios=((0.5,1.0,2.0),))
    model = FasterRCNN(backbone,num_classes=10,rpn_anchor_generator=anchorGen)

    checkpoint = torch.load(state_dict_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint

    model.eval()
    model.to(device)

    with torch.no_grad():
        metric_dict = {}
        for ii, data in enumerate(testLoader):
            sys.stdout.write("\r Completed {}/{}".format(ii,testDataset.__len__()))
            newImgs = []
            for imgTS in data['image']:
                imgTS = imgTS.float()
                newImgs.append(imgTS.to(device))
            output = model(newImgs)

            for pred, gt in zip(output, data['labels']):
                sub_metric_dict = {}
                boxCoord = pred['boxes'].cpu().detach().numpy()
                label = pred['labels'].cpu().detach().numpy()
                scores = pred['scores'].cpu().detach().numpy()

                coord_label_pred = []
                for cl, bC, sc in zip(label, boxCoord, scores):
                    if sc > threshold:
                        bC = list(bC)
                        bC.append(float(cl))
                        coord_label_pred.append(bC)


                for val in coord_label_pred:
                    if val[-1] not in gt[:, -1]:
                        sub_metric_dict[val[-1]] = 0.0
                    else:
                        maxIoU = 0.0
                        sub_organs = gt[gt[:, -1] == val[-1]]
                        for jj in range(sub_organs.shape[0]):
                            iou = IoU(sub_organs[jj, :], val[:-1])
                            if iou >= maxIoU:
                                maxIoU = iou
                        sub_metric_dict[val[-1]] = maxIoU

                metric_dict[ii] = sub_metric_dict

    recallCurve, iouRange = recall_curve(metric_dict)

    df = {"IoU":iouRange,"Recall":recallCurve,"Class":["Recall (All Classes)"]*iouRange.shape[0]}
    df = pandas.DataFrame(df)

    sns.set_context("talk")
    sns.lineplot(data=df,x="IoU",y="Recall")
    plt.show()

if __name__ == "__main__":
    test_args()
