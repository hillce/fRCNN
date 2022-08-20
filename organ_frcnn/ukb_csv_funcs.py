import csv
import os
import tempfile
import typing as t

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pydicom

from config import colorDict, textDict

#fieldNames = ['File','Organs Found','Scores','Coordinates']

def csvRead(csvObj: csv.DictReader) -> t.Dict:
    """Wrapper to read in all the lines of a csv containing the column 'File' into a dictionary

    Args:
        csvObj (csv.DictReader): csv object to iterate through

    Returns:
        csvDict (t.Dict): the dictionary split by file. 
    """
    csvDict = {}
    for row in csvObj:
        csvDict[row['File']] = row
    return csvDict


def plot_bounding_boxes(img: np.ndarray) -> np.ndarray:
    """Function to plot a bounding box on an image, and return the image

    Args:
        img (np.ndarray): The image to annotate.

    Returns:
        annotatedImage (np.ndarray): The image with bounding boxes plotted on them
    """
    colorScheme = {v:colorDict[k] for k,v in textDict.items()}

    annotatedImage = img
    fig, ax = plt.subplots(1)
    ax.imshow(annotatedImage)
    for coord,lab,sc in zip(csvDict['Coordinates'],csvDict['Organs Found'],csvDict['Scores']):
        rect = patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],linewidth=1,edgecolor=colorScheme[lab],facecolor='none')
        ax.add_patch(rect)

    with tempfile.TemporaryFile as tmp:
        fig.savefig(tmp.name)
        plt.close()

        annotatedImage = mpimg.imread(tmp.name)

    return annotatedImage


def load_image(
    subName: str,
    direc: str,
    grayscale: bool = False
) -> np.ndarray:
    """Loads the RGB version of a shMOLLI acquisition from UKBB, using instance times

    Args:
        subName (str): dicom folder name in the data directory
        direc (str): path to the dicom directory
        grayscale (bool, optional): Whether to just return a grayscale version of the image. Defaults to False.

    Returns:
        img (np.ndarray): The numpy ndarray of the image
    """
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

# if __name__ == "__main__":
#     direc = 'C:/Users/shug4421/UKB_Liver/shMOLLI_10765/Data'
#     fChoice = '4431819_20204_2_0'
#     gs = True
#     with open('biobankBoundingBoxes.csv','r') as f:
#         csvObj = csv.DictReader(f)
#         print("CSV opened with parameters {}".format(csvObj.fieldnames))
#         csvDict = csvRead(csvObj)
    

#     fChoiceBoxes = csvDict[fChoice]
#     img = load_image(fChoice,direc,gs=gs)
#     annotatedImage = plot_bounding_boxes(img,fChoiceBoxes)

#     plt.figure()
#     plt.imshow(annotatedImage)
#     plt.show()
