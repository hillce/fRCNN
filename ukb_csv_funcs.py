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
import matplotlib.image as mpimg
from PIL import Image
import pydicom

#fieldNames = ['File','Organs Found','Scores','Coordinates']

def csvRead(csvObj,dcmChoice=None):
    #fieldNames = csvObj.fieldnames
    csvDict = {}
    for row in csvObj:
        csvDict[row['File']] = row
    return csvDict

def plot_bounding_boxes(img,csvSubDict):
    colorScheme = {'Body':'r','Liver':'g','Cyst':'m','Lungs':'y','Heart':'c'}

    annotatedImage = img
    fig, ax = plt.subplots(1)
    ax.imshow(annotatedImage)
    orgFound = []
    for coord,lab,sc in zip(csvDict['Coordinates'],csvDict['Organs Found'],csvDict['Scores']):
        rect = patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],linewidth=1,edgecolor=colorScheme[lab],facecolor='none')
        ax.add_patch(rect)
    fig.savefig('temp_img.png')
    plt.close()

    annotatedImage = mpimg.imread('temp_img.png')
    return annotatedImage

def load_image(subName,direc,gs=True):
    dcmList = [os.path.join(direc,subName,x) for x in os.listdir(os.path.join(direc,subName))]
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

    if gs:
        img = img[:,:,1]

    return img

if __name__ == "__main__":
    direc = 'C:/Users/shug4421/UKB_Liver/shMOLLI_10765/Data'
    fChoice = '4431819_20204_2_0'
    gs = True
    with open('biobankBoundingBoxes.csv','r') as f:
        csvObj = csv.DictReader(f)
        print("CSV opened with parameters {}".format(csvObj.fieldnames))
        csvDict = csvRead(csvObj)
    

    fChoiceBoxes = csvDict[fChoice]
    img = load_image(fChoice,direc,gs=gs)
    annotatedImage = plot_bounding_boxes(img,fChoiceBoxes)

    plt.figure()
    plt.imshow(annotatedImage)
    plt.show()