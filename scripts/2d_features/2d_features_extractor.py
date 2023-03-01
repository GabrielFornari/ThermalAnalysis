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
        fileName = "img_" + str(iIdx) + ".png"
        img = cv.imread(folder + fileName)
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
        if fileName.endswith(".png"):
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

################################
## Testing functions

imgFeatures = imfeatures("D:/Flir E5x Data/Images/N14ZP7_noError_imgs/", cpu)

# Shooting frequency is 9 fps
timeStamp = np.r_[0:1339] / 9.0

plt.plot(timeStamp, [img['avg']/ 255 * 30 + 20 for img in imgFeatures])
plt.show()

################################
imgFeatures = imfeatures("D:/Flir E5x Data/Images/N14ZP7_noError_imgs/", mem_r_1)

plt.plot(timeStamp, [img['avg']/ 255 * 30 + 20 for img in imgFeatures])
plt.show()
