
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt


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
            fileName = folder + "img_" + str(iIdx) + ".png"
            img = cv.imread(fileName, cv.IMREAD_COLOR)
            #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            if img is not None:
                images.append(img)
    return images

def my_pngcount_from(folder):
    nFiles = 0
    for fileName in os.listdir(folder):
        if fileName.endswith(".png"):
            nFiles += 1
    return nFiles

if __name__ == "__main__":
    folderInput = "D:/Flir E5x Data/PNG Images/N14ZP7_IoTest_Crop/"
    nImgs = my_pngcount_from(folderInput)
    imgs = my_pngread_from(folderInput, range(5, nImgs, 12))
    folderOutput = "D:/Flir E5x Data/Giff/N14ZP7_IoTest/"

    idx = 1
    for img in imgs:
        cv.imwrite(folderOutput + "img_" + str(idx) + ".png", img)
        idx += 1
