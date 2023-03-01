
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

########################
########################

def my_imcount_from(folder):
    nFiles = 0
    for fileName in os.listdir(folder):
        if fileName.endswith(".tif"):
            nFiles += 1
    return nFiles

def my_imread_from(folder, idx = []):
    images = []
    nFiles = my_imcount_from(folder)
    if not idx:
        for iFile in range(1, nFiles+1):
            fileName = folder + "img_" + str(iFile) + ".tif"
            img = cv.imread(fileName, flags=(cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH))
            if img is not None:
                images.append(img)
    else:
        for iIdx in idx:
            fileName = folder + "img_" + str(iIdx) + ".tif"
            img = cv.imread(fileName, flags=(cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH))
            if img is not None:
                images.append(img)
    return images

def my_pngcount_from(folder):
    nFiles = 0
    for fileName in os.listdir(folder):
        if fileName.endswith(".png"):
            nFiles += 1
    return nFiles

def my_pngread_from(folder, idx = []):
    images = []
    nFiles = my_pngcount_from(folder)
    if not idx:
        for iFile in range(1, nFiles+1):
            fileName = folder + "img_" + str(iFile) + ".png"
            img = cv.imread(fileName, cv.IMREAD_COLOR)
            #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            if img is not None:
                images.append(img)
    else:
        for iIdx in idx:
            fileName = folder + "img_" + str(iFile) + ".png"
            img = cv.imread(fileName, cv.IMREAD_COLOR)
            #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            if img is not None:
                images.append(img)
    return images

########################
########################

# folderPath = "D:\Flir E5x Data\Tiff (32 bit)/N14ZP7_MemTest_2/"
folderPath = "D:/Flir E5x Data/Images/N14ZP7_MemTest_2/"

topBorderCoord = 7
leftBorderCoord = 8

height = 101
width = 148

#imgs = my_imread_from(folderPath)
imgs = my_pngread_from(folderPath)

#imgs = my_pngread_from(folderPath)
idx = 1
for iImg in imgs:
    croppedImg = iImg[topBorderCoord:topBorderCoord+height, leftBorderCoord:leftBorderCoord+width]
    cv.imwrite(folderPath[:-1] + "_Crop/img_" + str(idx) + ".png", croppedImg)
    idx += 1
