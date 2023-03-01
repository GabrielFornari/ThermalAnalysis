import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt


def imfeatures(folder, roi, imgIdxNames = []):
    nFiles = imcount(folder)
    if not imgIdxNames:
        imgIdxNames = [i for i in range(1, nFiles+1)] # Image names start from 1
        imgFeatures = [{} for i in range(nFiles)]
    else:
        imgFeatures = [{} for i in range(len(imgIdxNames))]

    iImg = 0
    for iIdx in imgIdxNames:
        fileName = "img_" + str(iIdx) + ".tif"
        return cv.imread(fileName, flags=(cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH))
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

def my_tifread(folder, idx):
    fileName = folder + "img_" + str(idx) + ".tif"
    return cv.imread(fileName, flags=(cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH))

def my_pngread(folder, idx):
    fileName = folder + "img_" + str(idx) + ".png"
    img = cv.imread(fileName, cv.IMREAD_COLOR)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def my_imRect(img, rectList):
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

def my_pngCrop(img):
    top = 7
    left = 8
    height = 101
    width = 148
    return img[top:top+height, left:left+width]

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

def my_imnorm(img):
    _imgMin = 20
    _imgMax = 50
    return (img.copy() - _imgMin) / _imgMax


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
cpu = [64, 41, 24, 24]
ssd = [86, 21, 18, 16]
pwr = [106, 46, 11, 11]
bios = [71,25,6,8]

usb = [111+2, 89+2, 8, 7]
audio = [49,8-1,6,6]
wlan = [28-2,8-1,6,6]

# if file is SimpleTest_#, make x+=1 and y-=2 (or -1)
video_1 = [105+1, 1-1, 3, 3]
video_2 = [110+1, 5-2, 5, 5]
video_3 = [112+1, 10-2, 3, 3]

mem_l_1 = [19-2, 26, 6, 5]
mem_l_2 = [19-2, 43, 6, 5]
mem_l_3 = [19-2, 62+1, 6, 5]
mem_l_4 = [19-2, 79+2, 6, 5]

#mem_r_1 = [41+7, 2+24, 6, 5]
#mem_r_2 = [41+7, 19+24, 6, 5]
#mem_r_3 = [41+7, 38+24, 6, 5]
#mem_r_4 = [41+7, 55+24, 6, 5]

mem_r_1 = [48-1, 26, 6, 5]
mem_r_2 = [48-1, 43, 6, 5]
mem_r_3 = [48-1, 62+1, 6, 5]
mem_r_4 = [48-1, 79+2, 6, 5]


################################
## Testing functions
################################

img = my_tifread("D:\Flir E5x Data\Tiff (32 bit)/N14ZP7_SimpleTest_Crop/", 500)
png_img = my_pngread("D:\Flir E5x Data\Images/N14ZP7_SimpleTest_2_Crop/", 700)

#png_img = my_pngCrop(png_img)

my_imRect(img, [mem_l_1, mem_l_2, mem_l_3, mem_l_4, mem_r_1, mem_r_2, mem_r_3, mem_r_4])
my_pngRect(png_img, [mem_l_1, mem_l_2, mem_l_3, mem_l_4, mem_r_1, mem_r_2, mem_r_3, mem_r_4])

my_imRect(img, [cpu, ssd, pwr, usb, wlan, bios, audio])
my_pngRect(png_img, [cpu, ssd, pwr, usb, wlan, bios, audio])

my_imRect(img, [video_1, video_2, video_3])
my_pngRect(png_img, [video_1, video_2, video_3])

plt.show()

#imgFeatures = imfeatures("D:/Flir E5x Data/Images/N14ZP7_noError_imgs/", cpu)

# Shooting frequency is 9 fps
#timeStamp = np.r_[0:1339] / 9.0

#plt.plot(timeStamp, [img['avg']/ 255 * 30 + 20 for img in imgFeatures])
#plt.show()
