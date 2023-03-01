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


# This function shows an image with a red vertical line at 'vPos'
def my_pltVLine(plt, xCoords):
    for xCoord in xCoords:
        plt.axvline(x = xCoord, color='r', linestyle='--')


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

# if file is SimpleTest_# or MemTest_#, make x+=2 and y+=2
usb = [111+2, 89+2, 8, 7]
# if file is SimpleTest_#, make y-=1
# if file is MemTest_#, make y-=2
audio = [49,8-1,6,6]
# if file is SimpleTest_#, make x-=2 and y-=1
# if file is MemTest_#, make x-=4 and y-=2
wlan = [28+2,8-2,6,6]

# if file is SimpleTest_# or MemTest_#, make x+=1 and y-=2 (or -1)
video_1 = [105, 1, 3, 3]
video_2 = [110, 5, 5, 5]
video_3 = [112, 10, 3, 3]

# if file is SimpleTest_# or MemTest_#, make x-=2 and y+=1 (or +2)
mem_l_1 = [19, 26, 6, 5]
mem_l_2 = [19, 43, 6, 5]
mem_l_3 = [19-2, 62+1, 6, 5]
mem_l_4 = [19-2, 79+2, 6, 5]

# if file is SimpleTest_# or MemTest_#, make x-=2 and y+=1 (or +2)
mem_r_1 = [41+7, 2+24, 6, 5]
mem_r_2 = [41+7, 19+24, 6, 5]
mem_r_3 = [41+7-2, 38+24+1, 6, 5]
mem_r_4 = [41+7-2, 55+24+2, 6, 5]

################################
## Testing functions
filePath = []
#filePath.append("D:/Flir E5x Data/Tiff (32 bit)/N14ZP7_SimpleTest_2_Crop/")
#filePath.append("D:/Flir E5x Data/Tiff (32 bit)/N14ZP7_SimpleTest_3_Crop/")
#filePath.append("D:/Flir E5x Data/Tiff (32 bit)/N14ZP7_SimpleTest_4_Crop/")

#filePath.append("D:/Flir E5x Data/Tiff (32 bit)/N14ZP7_SimpleTest_Crop/")
#filePath.append("D:/Flir E5x Data/Tiff (32 bit)/N14ZP7_MemTest_2_Crop/")
#filePath.append("D:/Flir E5x Data/Tiff (32 bit)/N14ZP7_ProcTest_Crop/")
#filePath.append("D:/Flir E5x Data/Tiff (32 bit)/N14ZP7_IoTest_Crop/")

#filePath.append("D:/Flir E5x Data/Tiff (32 bit)/N14ZP7_CPUError_Crop/")
filePath.append("D:/Flir E5x Data/Tiff (32 bit)/N14ZP7_MemError_Bios_Crop/")
#filePath.append("D:/Flir E5x Data/Tiff (32 bit)/N14ZP7_SimpleTest_2_Crop/")

selectedFeature = "avg"
#selectedROIs = [cpu, video_1, video_2, video_3]
selectedROIs = [cpu, bios, usb, audio, wlan]
#selectedROIs = [cpu, pwr, ssd]
#selectedROIs = [cpu, mem_l_1, mem_l_2, mem_l_3, mem_l_4, mem_r_1, mem_r_2, mem_r_3, mem_r_4]

#selectedROIs = [cpu, pwr, ssd]
#selectedROIs = [mem_l_1, mem_r_1]
#selectedROIs = [bios, wlan]
#selectedROIs = [mem_l_1, mem_l_2, mem_l_3, mem_l_4, mem_r_1, mem_r_2, mem_r_3, mem_r_4]


startingFrame = []
#startingFrame.append(round(23.9*9)) # SimpleTest_2
#startingFrame.append(round(23.3*9)) # SimpleTest_3
#startingFrame.append(round(25.55*9)) # SimpleTest_4

#startingFrame.append(round(21*9)) # SimpleTest
#startingFrame.append(round(25.7*9)) # MemTest_2
#startingFrame.append(round(26.35*9)) # ProcTest
#startingFrame.append(round(21.45*9)) # IoTest

startingFrame.append(round(101+30)) # N14ZP7_CPUError
startingFrame.append(round(26.6*9)) # N14ZP7_MemError_Bios
startingFrame.append(round(23.9*9)) # SimpleTest_2

startingPos = 100

frameIndices = []
#nFrames = []
for i in range(len(filePath)):
    #nFrames.append(imcount(filePath[i]))
    frameIndices.append(np.r_[startingFrame[i]-startingPos:imcount(filePath[i])])

for selectedROI in selectedROIs:
    imgFeatures = []
    for i in range(len(filePath)):
        imgFeatures.append(imfeatures(filePath[i], selectedROI, frameIndices[i]))
    avgValue = []
    for i in range(len(filePath)):
        avgValue.append([img[selectedFeature] + i*5 for img in imgFeatures[i]])
    # Shooting frequency is 9 fps
    for i in range(len(filePath)):
        plt.plot([iFrame / 9.0 for iFrame in range(len(frameIndices[i]))], avgValue[i], label = "Teste #" + str(i+1))


# SimpleTest_2, # SimpleTest_3, # SimpleTest_4
videoOnPos = startingPos + round(31*9)
batchOnPos = videoOnPos + round(40*9)
batchOffPos = batchOnPos + round(200*9)
screenOffPos = batchOffPos + round(30.6*9)

# SimpleTest
#videoOnPos = startingPos + round(24.2*9)
#batchOnPos = videoOnPos + round(20.24*9)
#batchOffPos = batchOnPos + round(200*9)
#screenOffPos = batchOffPos + round(30.95*9)

my_pltVLine(plt, [startingPos / 9.0, videoOnPos / 9.0, batchOnPos / 9.0 + 10, batchOffPos / 9.0 - 10, screenOffPos / 9.0])
#my_pltVLine(plt, [startingPos / 9.0, videoOnPos / 9.0, batchOnPos / 9.0 + 10, batchOffPos / 9.0 - 10])

# MemTest_2
#my_pltVLine(plt, [(21+23+20+131-10), (21+23+20+131)])

plt.grid(linestyle = '--', linewidth = 0.5)
plt.xlim(0, 350)
#plt.ylim(20, 40)

plt.xlabel("Tempo (s)")
plt.ylabel("Temperatura (ºC)")
plt.legend(loc="upper right")

plt.show()
