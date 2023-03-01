import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

# This function reads an image from a given path
def my_imread(path, flag=cv.IMREAD_COLOR):
    img = cv.imread(path, flag)
    if img is not None:
        return img
    else:
        raise RuntimeError("Couldn't read the image. Verify the image path.")


def my_imcount_from(folder):
    nFiles = 0
    for fileName in os.listdir(folder):
        if fileName.endswith(".png"):
            nFiles += 1
    return nFiles

def my_imread_from(folder, idx = []):
    images = []
    nFiles = my_imcount_from(folder)
    if not idx:
        for iFile in range(1, nFiles+1):
            fileName = "img_" + str(iFile) + ".png"
            img = cv.imread(folder + fileName)
            if img is not None:
                images.append(img)
    else:
        for iIdx in idx:
            fileName = "img_" + str(iIdx) + ".png"
            img = cv.imread(folder + fileName)
            if img is not None:
                images.append(img)
    return images

def my_file_name_from(folder):
    for fileName in os.listdir(folder):
        print(fileName)

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

def my_img_features(img, rect):
    features = []
    sub_img = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

    features.append(sub_img.min())
    features.append(sub_img.max())
    features.append(sub_img.mean())
    features.append(sub_img.std())

    return features

# Coordinates: x, y, w, h
cpu = [280, 162, 64, 64]
cpu_t = [336, 108, 56, 40]
cpu_r = [382, 188, 12, 8]
cpu_r2 = [398, 177, 25, 25]

top_1 = [404, 70, 9, 9]
top_2 = [386, 60, 10, 7]
top_3 = [410, 80, 7, 10]

mem_t = [184, 80, 20, 20]
mem_l_1 = [162, 126, 24, 18]
mem_l_2 = [160, 166, 24, 18]
mem_l_3 = [158, 218, 24, 18]
mem_l_4 = [156, 262, 24, 18]

mem_r_1 = [232, 124, 24, 18]
mem_r_2 = [232, 164, 24, 18]
mem_r_3 = [232, 216, 24, 18]
mem_r_4 = [234, 260, 24, 18]


imgs = my_imread_from("D:/Flir E5x Data/Images/N14ZP7_noError_imgs2/")

imgFeatures = []
for iImg in imgs:
    imgFeatures.append(my_img_features(iImg, cpu))

avgFeature = []
maxFeature = []
minFeature = []
for iFeature in imgFeatures:
    minFeature.append(iFeature[0])
    maxFeature.append(iFeature[1])
    avgFeature.append(iFeature[2]/255*30 + 20)

# Frequency of shooting is 9 fps
timeStamp = np.r_[0:1339] / 9.0

plt.plot(timeStamp, avgFeature)
plt.show()
