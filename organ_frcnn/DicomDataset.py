#!/usr/bin/python3

import os
import typing as t

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
from skimage import transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import classDict, imgMean, imgStd


class DicomDataset(Dataset):
    """ Dicom Dataset."""

    def __init__(
        self,
        csvFile: str,
        rootDir: str,
        transform=None,
        img_dtype=np.float16):
        """
        Args:
            csvFile (str): Path to data.csv with name and x,y coordinates
            rootDir (str): Path to directory for all the dicoms
            transform (callable, optional): Optional transform to be applied on a sample.
            img_dtype (type): the dtype to be applied to the image.

        Example Usage:

            Folder structure:

                /path/to/dicom/folders
                            |
                            - DICOM FOLDER NAME 1
                                        |
                                        - 0001.dcm
                                        - 0002.dcm
                                        ....
                            - DICOM FOLDER NAME 2
                            ...


            CSV structure:

                train.cav

                DICOM FOLDER NAME 1, CLASS 1, x0, y0, x1, y1, CLASS 2, ... etc.
                DICOM FOLDER NAME 2, CLASS 1, x0, y0, x1, y1, CLASS 4, ... etc.
                ...

                val.csv
                test.csv

            CALL:
                DicomDataset("train.csv", "/path/to/dicom/folders/, transform=train_transforms)


        """
        self.data = pd.read_csv(csvFile,header=None)

        self.rootDir = rootDir
        self.transform = transform
        self.classDict = classDict
        self.img_dtype = img_dtype

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dcmName = os.path.join(self.rootDir,self.data.iloc[idx,0])
        dcmList = [os.path.join(dcmName,x) for x in os.listdir(dcmName) if x.endswith('.dcm')]

        instTime = {} # Sort the images by instance time
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
        
        img = img[:,:,(2,1,0)].astype(self.img_dtype)


        labels = self.data.iloc[idx,1:]
        labels = np.array([labels])
        labels = labels.reshape(-1,5)

        for i in range(len(labels)):
            if pd.isna(labels[i,4]):
                break

            labels[i,4] = self.classDict[labels[i,4]]

        sample = {'image':img,'labels':labels}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def resize_img(self, img, width, height, resizeSize=400):
        """Function to resize the image.

        Args:
            img (np.ndarray): image to resize
            width (int): target width
            height (int): target height
            resizeSize (int, optional): maximal length. Defaults to 400.

        Returns:
            newImg (np.ndarray): resized image
        """
        if width <= height:
            f = float(resizeSize) / width
            resized_height = int(f * height)
            resized_width = int(resizeSize)
        else:
            f = float(resizeSize) / height
            resized_width = int(f * width)
            resized_height = int(resizeSize)

        newImg = cv2.resize(img, (resized_height, resized_width), interpolation=cv2.INTER_CUBIC)
        return newImg

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w), preserve_range=True)

        # for i in range(img.shape[2]):
        #     img[:,:,i] = 255*img[:,:,i]//np.amax(img[:,:,i])
        # img = img.astype(np.uint8)
        # #img = cv2.resize(image,(new_h,new_w),interpolation=cv2.INTER_CUBIC)
        # # h and w are swapped for landmarks because for images,
        # # x and y axes are axis 1 and 0 respectively
        
        newLabels = []
        for label in labels:
            if np.isnan(label[0]):
                break
            tempLabel = label[:4] * [new_w / w, new_h / h, new_w / w, new_h / h]
            tempLabel = list(tempLabel)
            tempLabel.append(label[4])
            newLabels.append(tempLabel)
        labels = np.array(newLabels)

        return {'image': img, 'labels': labels}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.    
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,left: left + new_w]
        
        newLabels = []
        for label in labels:
            if np.isnan(label[0]):
                break
            tempLabel = label[:4] - [left, top, left, top]
            tempLabel = list(tempLabel)
            tempLabel.append(label[4])
            newLabels.append(tempLabel)
        labels = np.array(newLabels)
        return {'image': image, 'labels': labels}

class ToTensor(object):
    """ convert ndarrays in sample to Tensors"""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        image = image.transpose((2,0,1))
        return {'image': torch.from_numpy(image),'labels': torch.from_numpy(labels)}

class Normalise(object):
    """Normalise input images with mean and std"""

    def __init__(self):
        self.normImg = transforms.Normalize(imgMean, imgStd, inplace=True)

    def __call__(self,sample):
        img = sample['image'].float()
        return {'image':self.normImg(img),'labels':sample['labels']}

def collate_var_rois(sampleBatch: t.List) -> t.Dict: #Collate Function for DataLoader
    """Collate Function for Data Loader

    Args:
        sampleBatch (t.List): List of dictionarys [tensors and labels]

    Returns:
        sample (t.Dict): Dictionary of lists [tensors and labels]
    """
    img = [item['image'] for item in sampleBatch]
    labels = [item['labels'] for item in sampleBatch]
    sample = {'image':img,'labels':labels}
    return sample
