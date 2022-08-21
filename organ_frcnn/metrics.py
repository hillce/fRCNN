import json
import re
import typing as t

import numpy as np
import pandas

from config import classes

# prediction_json = "Biobank_Bounding_Boxes_2021_9_6_mobilenet_v3.json"
# ground_truth_json = "UK Biobank large test n19695.json"

def load_json(
    json_path: str,
):
    with open(json_path, "r") as f:
        json_obj = json.load(f)

    return json_obj


def format_predictions(
    prediction_json_path: str,
    gt_ids: t.Union[t.List, t.Type[None]] = None,
    train_ids: str = "train_new_format.csv",
    val_ids: str = "val_new_format.csv" 
) -> t.Tuple[np.ndarray, t.List]:
    preds = load_json(prediction_json_path)

    if gt_ids is None:
        gt_ids = list(preds.keys())
        train_ids = pandas.read_csv(train_ids, header=None)
        val_ids = pandas.read_csv(val_ids, header=None)

        train_ids = list(train_ids[0])
        val_ids = list(val_ids[0])

        train_ids.extend(val_ids)

        gt_ids = [x for x in gt_ids if x not in train_ids]

    print("Number of GT ids = {}".format(len(gt_ids)))
    bbInput = np.zeros((len(gt_ids),(len(classes)+2)*4))

    for i,eid in enumerate(gt_ids):
        bBoxes = preds[eid]
        bBKeys = list(bBoxes.keys())
        j = 0
        for cl in classes:
            patt = str("{}\_\d+".format(cl))
            numOrgs = []
            
            for bb in bBKeys:
                org = re.findall(patt,bb)
                numOrgs.extend(org)

            predAcc = np.array([x[4] for x in numOrgs])
            numOrgs = np.array(numOrgs)

            if cl in ["Lung", "Kidney"]:
                topPred = predAcc.argsort()[-2:]
                tempKeys = numOrgs[topPred]
                tempBbox = np.array([bBoxes[tK][:4] for tK in tempKeys])
                tempBbox = tempBbox.flatten()
                bbInput[i,j:j+(4*len(tempKeys))] = tempBbox

                j += 8
            else:
                try:
                    topPred = predAcc.argsort()[-1:]
                    tempKey = numOrgs[topPred.astype(int)][0]
                    tempBbox = bBoxes[tempKey][:4]
                    bbInput[i,j:j+4] = tempBbox
                except IndexError:
                    pass

                j += 4

    return bbInput, gt_ids


def area(inp: np.ndarray) -> np.ndarray:
    vol = np.zeros((inp.shape[0],inp.shape[1]//4))
    for i,arr in enumerate(inp):
        for j in range(0,inp.shape[1],4):
            vol[i,j//4] = (arr[j+2]-arr[j])*(arr[j+3]-arr[j+1])
    return vol


def areas_to_dictionary(
    areas: np.ndarray,
    eids: t.List
) -> t.Dict:
    area_dict = {}
    for eid,arr in zip(eids,areas):
        tempDict = {}

        i = 0
        j = 0
        while i < arr.shape[0]:

            if i == 2 or i == 5:
                val = np.mean(arr[i:i+1])
                if val < 0.01:
                    val = np.nan
                tempDict[classes[j]] = val
                i += 2
            else:
                val = arr[i]
                if val < 0.01:
                    val = np.nan
                tempDict[classes[j]] = val
                i += 1

            j += 1

        area_dict[eid] = tempDict

    return area_dict


def area_to_body_ratio(
    area_dict: t.Dict
) -> pandas.DataFrame:
    trainRatioDict = {}
    for uid in area_dict.keys():
        tempDict = {}
        for org in classes:
            tempDict[org] = area_dict[uid][org]/area_dict[uid]["Body"]
        trainRatioDict[uid] = tempDict

    ratioDf = pandas.DataFrame(trainRatioDict)
    ratioDf = ratioDf.T

    return ratioDf


def IoU(coords0,coords1):
    bM0 = np.zeros((288,384))
    bM1 = np.zeros((288,384))

    for i in range(288):
        for j in range(384):
            if i >= coords0[1] and i < coords0[3]:
                if j >= coords0[0] and j < coords0[2]:
                    bM0[i,j] = 1
            if i >= coords1[1] and i < coords1[3]:
                if j >= coords1[0] and j < coords1[2]:
                    bM1[i,j] = 1

    intersect = np.logical_and(bM0,bM1)
    union = np.logical_or(bM0,bM1)  

    iou = np.sum(intersect)/np.sum(union)
    return iou


def recall_curve(iou_dict: t.Dict, cl: t.Union[t.Type[None], int] = None):
    iouRange = np.arange(0,1,0.01)
    iouRange = iouRange[1:]
    recallCurve = np.zeros_like(iouRange)

    for i,iou in enumerate(iouRange):
        tp = 0
        fn = 0

        for j, case_iou in iou_dict.items():
            for org in case_iou.keys():
                if cl:
                    if org == cl:
                        if case_iou[org] >= iou:
                            tp += 1
                        else:
                            fn += 1
                else:
                    if case_iou[org] >= iou:
                        tp += 1
                    else:
                        fn += 1

        recallCurve[i] = tp/(tp+fn)

    return recallCurve, iouRange