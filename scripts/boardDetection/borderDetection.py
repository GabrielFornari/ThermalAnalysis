import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

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

# This function shows an image with a red horizontal line at 'hPos'
def my_imHLine(img, hPos):
    (h, w) = img.shape[:2]
    if len(img.shape) == 2:
        tmpImg = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    else:
        tmpImg = img.copy()
    cv.line(tmpImg, (0,hPos), (w,hPos), (0,0,255), 1)
    plt.figure()
    plt.imshow(tmpImg)
    return tmpImg

# This function shows an image with a red vertical line at 'vPos'
def my_imVLine(img, vPos):
    (h, w) = img.shape[:2]
    if len(img.shape) == 2:
        tmpImg = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    else:
        tmpImg = img.copy()
    cv.line(tmpImg, (vPos,0), (vPos,h), (0,0,255), 1)
    plt.figure()
    plt.imshow(tmpImg)
    return tmpImg

def my_imnorm(img):
    _imgMin = 20
    _imgMax = 50
    return (img.copy() - _imgMin) / _imgMax

def my_imLine(img, coord1, coord2):
    (h, w) = img.shape[:2]
    if len(img.shape) == 2:
        tmpImg = my_imnorm(img)
        cv.cvtColor(tmpImg, cv.COLOR_GRAY2RGB)
    else:
        tmpImg = my_imnorm(img)
    cv.line(tmpImg, coord1, coord2, (0,0,255), 1)
    plt.figure()
    plt.imshow(tmpImg)
    return tmpImg

def my_imRect(img, rectPos, rectSize):
    (h, w) = img.shape[:2]
    if len(img.shape) == 2:
        tmpImg = my_imnorm(img)
        cv.cvtColor(tmpImg, cv.COLOR_GRAY2RGB)
    else:
        tmpImg = my_imnorm(img)
    cv.rectangle(tmpImg, rectPos, [rectPos[0] + rectSize[0], rectPos[1] + rectSize[1]], (0,0,255), 1)
    plt.figure()
    plt.imshow(tmpImg)
    return tmpImg

def my_imCrop(img):
    (h, w) = img.shape[:2]
    return img[:h-4,:w-1]

def my_imHIdx(img, hLineIdx, imgThreshold):
    (h, w) = img.shape[:2]
    # The border should be on the first 1/3 of the image
    imgHLine = img[hLineIdx, :round(w/3)]
    idx = [idx for idx, value in enumerate(imgHLine) if value >= imgThreshold]
    return idx[0]
    #firstIdx = [idx for idx, value in enumerate(imgHLine) if value >= imgThreshold]
    #imgHLine = img[hLineIdx, round(w/3):w]
    #lastIdx = [idx for idx, value in enumerate(imgHLine) if value < imgThreshold]
    #return firstIdx[0], lastIdx[0] + round(w/3) - 1


def my_imVIdx(img, vLineIdx, imgThreshold):
    (h, w) = img.shape[:2]
    # The border should be on the first 1/3 of the image
    imgVLine = img[:round(h/3), vLineIdx]
    idx = [idx for idx, value in enumerate(imgVLine) if value >= imgThreshold]
    return idx[0]
    #firstIdx = [idx for idx, value in enumerate(imgVLine) if value >= imgThreshold]
    #imgVLine = img[round(h/3):h, vLineIdx]
    #lastIdx = [idx for idx, value in enumerate(imgVLine) if value < imgThreshold]
    #return firstIdx[0], lastIdx[0] + round(h/3) - 1

# To do: create a function that shows the horizontal and vertical profiles
#        of the board

imgs = my_imread_from("D:\Flir E5x Data\Tiff (32 bit)/N14ZP7_MemTest_1/", [1500])

# Remove weird borders
#img = my_imCrop(imgs[0])
img = imgs[0]

hLine = 30
threshold = 26
my_imLine(img, [0, hLine], [160, hLine])
plt.figure()
plt.plot(range(len(img[hLine,:])), img[hLine,:])

leftBorderCoord = my_imHIdx(img, hLine, threshold)+1
my_imLine(img, [leftBorderCoord, 0], [leftBorderCoord, 120])

vLine = 138
threshold = 27
my_imLine(img, [vLine, 0], [vLine, 120])
plt.figure()
plt.plot(range(len(img[:, vLine])), img[:, vLine])

topBorderCoord = my_imVIdx(img, vLine, threshold)+2
my_imLine(img, [0, topBorderCoord], [160, topBorderCoord])


width = 148
height = 101

my_imRect(img, [leftBorderCoord, topBorderCoord], [width, height])
croppedImg = img[topBorderCoord:topBorderCoord+height, leftBorderCoord:leftBorderCoord+width]
my_imLine(croppedImg, [0, 100], [0, 100])

print("Top Border: ", topBorderCoord)
print("Left Border: ", leftBorderCoord)

print("Height: ", height)
print("Width: ", width)

plt.show()
