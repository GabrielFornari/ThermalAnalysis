import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

###############
## Functions ##
###############

def imrename(folder):
    nFiles = imcount(folder)
    imgIdxNames = [i for i in range(1, nFiles+1)] # Image names start from 1

    iImg = 0
    for iIdx in imgIdxNames:
        oldFileName = folder + "N14ZP7_SimpleTest_" + str(iIdx) + ".tif"
        newFileName = folder + "img_" + str(iIdx) + ".tif"
        os.rename(oldFileName, newFileName)

def imcount(folder):
    nFiles = 0
    for fileName in os.listdir(folder):
        if fileName.endswith(".tif"):
            nFiles += 1
    return nFiles

# This function shows an image in a defined size
def my_imshow(img, width=600, height=400, inter=cv.INTER_AREA):
    dim = None
    (h, w) = img.shape[:2]

    if height is not None:
        r = height / float(h)
        dim = (int(w * r), height)
    if width is not None:
        r = width / float(w)
        dim = (width, int(h * r))

    tmpImg = cv.resize(img, dim, interpolation=inter)
    cv.imshow('', tmpImg)
    cv.waitKey(0)

#######################
## Testing functions ##
#######################

imrename("D:/Flir E5x Data/Tiff (32 bit)/N14ZP7_SimpleTest/")
