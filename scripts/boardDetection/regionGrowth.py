import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

##############################
##############################

def my_tifread(folder, idx):
    fileName = folder + "img_" + str(idx) + ".tif"
    return cv.imread(fileName, flags=(cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH))

def my_pngread(folder, idx):
    fileName = folder + "img_" + str(idx) + ".png"
    img = cv.imread(fileName, cv.IMREAD_COLOR)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def my_tifRect(img, rectList):
    (h, w) = img.shape[:2]
    if len(img.shape) == 2:
        tmpImg = my_imnorm(img)
        cv.cvtColor(tmpImg, cv.COLOR_GRAY2RGB)
    else:
        tmpImg = my_imnorm(img)

    for rect in rectList:
        cv.rectangle(tmpImg, [rect[0], rect[1]], [rect[0]+rect[2], rect[1]+rect[3]], (0,0,255), 1)

    plt.figure()
    plt.imshow(tmpImg)
    return tmpImg

def my_imnorm(img):
    _imgMin = 20
    _imgMax = 50
    return (img.copy() - _imgMin) / _imgMax

def my_pngRect(img, rectList):
    (h, w) = img.shape[:2]
    if len(img.shape) == 2:
        cv.cvtColor(tmpImg, cv.COLOR_GRAY2RGB)

    tmpImg = img.copy()

    for rect in rectList:
        cv.rectangle(tmpImg, [rect[0], rect[1]], [rect[0]+rect[2], rect[1]+rect[3]], (0,0,255), 1)

    plt.figure()
    plt.imshow(tmpImg)
    return tmpImg


def my_regionGrowth(img, idx2visit, threshold):
    idxVisited = []
    (height, width) = img.shape[:2]
    while idx2visit:
        idx = idx2visit.pop()
        if [idx[0], idx[1]] not in idxVisited:
            if img[idx[0], idx[1]] > threshold:
                idxVisited.append([idx[0], idx[1]])

                # add neighbors
                if idx[0]-1 >= 0:
                    idx2visit.append([idx[0]-1, idx[1]])
                if idx[1]-1 >= 0:
                    idx2visit.append([idx[0], idx[1]-1])
                if idx[0]+1 < height:
                    idx2visit.append([idx[0]+1, idx[1]])
                if idx[1]+1 < width:
                    idx2visit.append([idx[0], idx[1]+1])

    return idxVisited

def my_pngCrop(img):
    top = 4
    left = 9
    height = 101
    width = 148
    return img[top:top+height, left:left+width]

##############################
##############################

imgTest1 = [[0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 8, 7, 7],
            [0, 0, 0, 7, 9]]

idx2visit = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]

imgTest1 = np.pad(imgTest1, (0,0), 'constant')
print(imgTest1)

print(imgTest1[1,1])

value = 7;
while idx2visit:
    idx = idx2visit.pop()
    print(idx)
    if idx == [10, 10]:
        break;
    else:
        idx2visit.append([value, value])
    value += 1

print(idx2visit)

print(imgTest1.shape)


regionTest1 = my_regionGrowth(imgTest1, [[3,4]], 5)

print(regionTest1)
