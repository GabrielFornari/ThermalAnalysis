import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# This function reads an image from a given path
def my_imread(path, flag=cv.IMREAD_COLOR):
    img = cv.imread(path, flag)
    if img is not None:
        return img
    else:
        raise RuntimeError("Couldn't read the image. Verify the image path.")

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


img = my_imread("D:/Gabriel/14. Cepedi/Python/thermal/FLIR0041 - Copy - Copy.jpg")
#img = my_imread("D:/Gabriel/14. Cepedi/Python/thermal/FLIR0041.jpg")

print(img.shape)
#print(img[0:10, 0:10, 2])
my_imshow(img[:,:,:])

subimg = img[110:220,20:100,:]
my_imshow(subimg)
print(subimg.max())
print(subimg.max()*25/255 + 20)

subimg = img[160:220,220:300,1]
my_imshow(subimg)
print(subimg.max())
print(subimg.max()*25/255 + 20)
print(subimg.min())
print(subimg.min()*25/255 + 20)
