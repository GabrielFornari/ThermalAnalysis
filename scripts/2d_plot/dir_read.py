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



imgs = my_imread_from("D:/Flir E5x Data/Images/N14ZP7_noError_imgs2/", [700])

img = imgs[0]

print(str(np.size(img, 0)) + " x " + str(np.size(img,1)))
my_imshow(img)
