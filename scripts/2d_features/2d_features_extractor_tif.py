import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt


def imfeatures1(folder, roi, imgIdxNames = []):
    nFiles = imcount(folder)
    if not imgIdxNames:
        imgIdxNames = [i for i in range(0, nFiles+0)] # Image names start from 1
        imgFeatures = [{} for i in range(nFiles)]
    else:
        imgFeatures = [{} for i in range(len(imgIdxNames))]

    iImg = 0
    for iIdx in imgIdxNames:
        fileName = folder + "N14ZP7_CPUError_OnOff_" + str(iIdx) + ".tif"
        img = cv.imread(fileName, flags=(cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH))
        if img is not None:
            # for each ROI:
            imgFeatures[iImg] = extractfeatures(img, roi)
        else:
            print("File '" + fileName + "' not found.")
        iImg += 1
    return imgFeatures

def imfeatures(folder, roi, imgIdxNames = []):
    nFiles = imcount(folder)
    if not imgIdxNames:
        imgIdxNames = [i for i in range(0, nFiles+0)] # Image names start from 1
        imgFeatures = [{} for i in range(nFiles)]
    else:
        imgFeatures = [{} for i in range(len(imgIdxNames))]

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


# ROI
# Coordinates: x, y, w, h
cpu = [75, 55, 10, 10]

################################
## Testing functions

imgFeatures = imfeatures("D:/Flir E5x Data/Tiff (32 bit)/N14ZP7_CPUError_test/", cpu)
imgFeatures1 = imfeatures1("D:/Flir E5x Data/Tiff (32 bit)/N14ZP7_CPUError/", cpu)

# Shooting frequency is 9 fps
timeStamp = np.r_[0:2380] / 9.0

avgValue = [img['avg'] for img in imgFeatures]
avgValue1 = [img['avg'] for img in imgFeatures1]

plt.plot(timeStamp, avgValue)
plt.show()
plt.plot(timeStamp, avgValue1)
plt.show()

zip_object = zip(avgValue, avgValue1)
difference = []
for avgValue, avgValue1 in zip_object:
    difference.append(avgValue - avgValue1)

plt.plot(timeStamp, difference)
plt.show()
