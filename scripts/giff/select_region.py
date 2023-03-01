import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

##############################
##############################

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
            if img[idx[1], idx[0]] > threshold:  # x and y axix are inverted in imgs
                idxVisited.append([idx[0], idx[1]])

                # add neighbors
                if idx[0]-1 >= 0:
                    idx2visit.append([idx[0]-1, idx[1]])
                if idx[1]-1 >= 0:
                    idx2visit.append([idx[0], idx[1]-1])
                if idx[0]+1 < width:
                    idx2visit.append([idx[0]+1, idx[1]])
                if idx[1]+1 < height:
                    idx2visit.append([idx[0], idx[1]+1])

    return idxVisited

def my_pngROI(img, idxVisited):
    ROIImg = img.copy()
    (height, width) = ROIImg.shape[:2]
    for x in range(width):
        for y in range(height):
            if [x, y] not in idxVisited:
                ROIImg[y, x, :] = [255, 255, 255] # white

    return ROIImg

def my_pngROICrop(img, idxVisited, gap = 10):
    ROIImg = my_pngROI(img, idxVisited)

    minX = min(selectedROI, key=lambda x: x[0])[0]
    maxX = max(selectedROI, key=lambda x: x[0])[0]
    minY = min(selectedROI, key=lambda x: x[1])[1]
    maxY = max(selectedROI, key=lambda x: x[1])[1]

    newImage = np.zeros((maxY-minY+gap*2,maxX-minX+gap*2,3), np.uint8)
    newImage[:, :] = [255, 255, 255]
    newImage[gap:-gap, gap:-gap] = ROIImg[minY:maxY, minX:maxX]

    return newImage


def my_tifread(folder, idx):
    fileName = folder + "img_" + str(idx) + ".tif"
    return cv.imread(fileName, flags=(cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH))

def my_pngread(folder, idx):
    fileName = folder + "img_" + str(idx) + ".png"
    img = cv.imread(fileName, cv.IMREAD_COLOR)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

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

def my_imRect(img, rectList = []):
    (h, w) = img.shape[:2]
    if len(img.shape) == 2:
        #tmpImg = my_imnorm(img)
        tmpImg = img.copy()
        cv.cvtColor(tmpImg, cv.COLOR_GRAY2RGB)
    else:
        #tmpImg = my_imnorm(img)
        tmpImg = img.copy()

    for rect in rectList:
        cv.rectangle(tmpImg, [rect[0], rect[1]], [rect[0]+rect[2], rect[1]+rect[3]], (0,0,255), 1)

    plt.figure()
    plt.imshow(tmpImg)
    return tmpImg


def my_getMaxIndices(img, rect):
    tmpImg = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    [max_y, max_x] = np.unravel_index(np.argmax(tmpImg, axis=None), tmpImg.shape)
    return [max_x+rect[0], max_y+rect[1]]

# ROI
# Coordinates: x, y, w, h
cpu = [64, 41, 24, 24]
ssd = [86, 21, 18, 16]
pwr = [106, 46, 11, 11]
usb = [111, 89, 8, 7]
bios = [71,25,6,8]
audio = [49,8,6,6]
wlan = [28,8,6,6]

# if file is SimpleTest_#, make x+=1 and y-=2 (or -1)
video_1 = [105, 1, 3, 3]
video_2 = [110, 5, 5, 5]
video_3 = [112, 10, 3, 3]

mem_l_1 = [19, 26, 6, 5]
mem_l_2 = [19, 43, 6, 5]
mem_l_3 = [19, 62, 6, 5]
mem_l_4 = [19, 79, 6, 5]

mem_r_1 = [41+7, 2+24, 6, 5]
mem_r_2 = [41+7, 19+24, 6, 5]
mem_r_3 = [41+7, 38+24, 6, 5]
mem_r_4 = [41+7, 55+24, 6, 5]

##############################
##############################

tifImg = my_tifread("D:/Flir E5x Data/Tiff (32 bit)/N14ZP7_IoTest_Crop/", 1200)

ROIIdx = my_getMaxIndices(tifImg, cpu)
threshold = 38.5

selectedROI = my_regionGrowth(tifImg, [ROIIdx], threshold)

nImgs = my_pngcount_from("D:/Flir E5x Data/PNG Images/N14ZP7_IoTest_Crop/")
imgs = my_pngread_from("D:/Flir E5x Data/PNG Images/N14ZP7_IoTest_Crop/", range(5, nImgs, 12))

folderOutput = "D:/Flir E5x Data/Giff/N14ZP7_IoTest_CPU/"

idx = 1
for img in imgs:
    cv.imwrite(folderOutput + "img_" + str(idx) + ".png", my_pngROICrop(img, selectedROI))
    idx += 1
