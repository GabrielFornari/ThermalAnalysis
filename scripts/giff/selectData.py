
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt


def imfeatures(folder, roi, imgIdxNames = []):
    nFiles = imcount(folder)
    if imgIdxNames.any():
        imgFeatures = [{} for i in range(len(imgIdxNames))]
    else:
        imgIdxNames = [i for i in range(1, nFiles+1)] # Image names start from 1
        imgFeatures = [{} for i in range(nFiles)]


    iImg = 0
    for iIdx in imgIdxNames:
        fileName = folder + "img_" + str(iIdx) + ".tif"
        img = cv.imread(fileName, flags=(cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH))
        if img is not None:
            # for each ROI:
            imgFeatures[iImg] = extractfeatures(img, roi)
        else:
            print("File '" + fileName + "' not found.")
        iImg += 1
    return imgFeatures

def imcount(folder):
    nFiles = 0
    for fileName in os.listdir(folder):
        if fileName.endswith(".tif"):
            nFiles += 1
    return nFiles

def extractfeatures(img, rect):
    sub_img = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

    features = {'min':0, 'max':0, 'avg':0, 'std': 0}
    features['min'] = sub_img.min()
    features['max'] = sub_img.max()
    features['avg'] = sub_img.mean()
    features['std'] = sub_img.std()

    return features

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

cpu = [64, 41, 24, 24]

if __name__ == "__main__":
    folderInput = "D:/Flir E5x Data/Tiff (32 bit)/N14ZP7_IoTest_Crop/"
    nImgs = my_imcount_from(folderInput)
#    frameIndices = range(5, nImgs, 12)

    frameIndices = np.r_[[i for i in range(5, nImgs, 12)]]
    #imgs = my_imread_from(folderInput, imgIndex)
    folderOutput = "D:/Flir E5x Data/Giff/N14ZP7_IoTest_TimeSeries/"

    print(frameIndices[0:10])

    features = imfeatures(folderInput, cpu, frameIndices)



    avgTemperature = []
    for ftr in features:
        avgTemperature.append(ftr["avg"])

    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.xlim(0, 350)
    plt.ylim(20, 50)

    plt.xlabel("Tempo (s)", fontsize=16)
    plt.ylabel("Temperatura (ºC)", fontsize=16)
    for idx in range(0, len(frameIndices)):
        plt.plot(frameIndices[0:idx] / 9.0, avgTemperature[0:idx], linewidth = 3, c = 'blue')
        #plt.show()
        plt.savefig(folderOutput + "img_" + str(idx+1) + ".png")
