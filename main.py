from PIL import Image 
import numpy as np
import os
from math import pow, sqrt
from hashlib import md5
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, Pipe
from json import dumps, loads


acceptedExt = ['jpg', 'jpeg','png','bmp', "tiff"]

def filter_files(filenames):
    acceptedFiles = []
    for fname in filenames:
        ext = (fname.split('.'))[-1]
        if ext in acceptedExt: 
            acceptedFiles.append(fname)
    return acceptedFiles

def normalCurveFunction(variance , segment):
    points = []
    y_sum = 0
    if variance == 0:
        return [],0
    for x in range(5, 96, 90//segment):
        y = pow(np.exp(1), - pow(x - 50, 2) / (2 * variance) ) /sqrt(2 * np.pi * variance)
        points.append((x,y))
        y_sum += y  
    return points, y_sum

def expCurveFunction(mean, segment):
      points = [] 
      y_sum = 0
      if segment == 0:
          return [],0
      for x in range(5,96,90//segment):
          y = mean * pow(np.exp(1), -(mean * x))
          points.append((x,y))
          y_sum += y
      return points,y_sum

def getPercentOfImageSize(image, percent):
    h, w = image.size
    return (int(h * (percent/100)), int(w * (percent/100)))

def identifyExactDuplicates(imageList):
    imageHashDict = {}
    for imageName in imageList:
        hash = md5(np.array(Image.open(imageName))).hexdigest()
        if hash in imageHashDict:
            imageHashDict[hash].append(imageName)
        else:
            imageHashDict[hash] = [imageName]
    duplicates = list(filter(lambda x: len(x) > 1, imageHashDict.values()))
    return duplicates
 
def preHammingPrep1(image1, image2):
    resizeFactor = 0.05
    lowerBoundSize = (min(image1.size[0], image2.size[0]), min(image1.size[1], image2.size[1]))
    image1 = image1.resize(lowerBoundSize, resample=Image.LANCZOS)
    image2 = image2.resize(lowerBoundSize, resample=Image.LANCZOS)
    image1 = image1.resize((int (lowerBoundSize[0]*resizeFactor), int (lowerBoundSize[1]*resizeFactor)), resample=Image.LANCZOS)
    image2 = image2.resize((int (lowerBoundSize[0]*resizeFactor), int (lowerBoundSize[1]*resizeFactor)), resample=Image.LANCZOS)
    return image1, image2

def preHammingPrep2(image1, image2, resizeFactor):
    image1 = image1.resize(getPercentOfImageSize(image1, resizeFactor), resample=Image.LANCZOS)
    image2 = image2.resize(getPercentOfImageSize(image2, resizeFactor), resample=Image.LANCZOS)
    image1 = image1.convert('I')
    image2 = image2.convert('I')
    npImage1 = np.array(image1)
    npImage2 = np.array(image2)
    return npImage1, npImage2

def calculateSimilarityScore(imName1, imName2, approach, approachControlParam, approachSamplingParam):
    print(imName1, imName2, approach, approachControlParam, approachSamplingParam)
    image1=Image.open(imName1)
    image2=Image.open(imName2)
    image1, image2 = preHammingPrep1(image1, image2)
    cumulativeSimilarityScore = 0
    if approach == 'N':
        samplePts, sampleSum = normalCurveFunction(approachControlParam, approachSamplingParam)
    elif approach == 'E':
        samplePts, sampleSum = expCurveFunction(approachControlParam, approachSamplingParam)
    resizeSimilarityPlot = []
    # print(samplePts)
    for (resizeFactor, factorWeightage) in samplePts:
        npImage1, npImage2 = preHammingPrep2(image1, image2, resizeFactor)
        npGradient1 = np.diff(npImage1) > 1
        npGradient2 = np.diff(npImage2) > 1
        currentSimilarityScore = (np.count_nonzero(np.logical_not(np.logical_xor(npGradient1, npGradient2)))/npGradient1.size)
        weightedSimilarityScore = factorWeightage * currentSimilarityScore
        resizeSimilarityPlot.append((resizeFactor, currentSimilarityScore))
        cumulativeSimilarityScore += weightedSimilarityScore
    averageSimilarityScore = (cumulativeSimilarityScore / sampleSum) * 100
    print("{:f} %".format(round(averageSimilarityScore, 3)))
    return resizeSimilarityPlot, averageSimilarityScore

os.chdir(os.getcwd() + '/assets')
imFileNames = filter_files(os.listdir())
print(imFileNames)



def processingSimilarityWithAllDirectoryImages(imFileNames, approach, approachControlParam, approachSamplingParam):
    imgCount = len(imFileNames)
    if imgCount < 1:
        return
    for im2 in range(1,imgCount):
        rsp, av = calculateSimilarityScore(imFileNames[0], imFileNames[im2], approach, approachControlParam, approachSamplingParam)
        #plt.plot((*zip(*rsp)))
        #plt.show()
        # print(rsp,av)
    return processingSimilarityWithAllDirectoryImages(imFileNames[1:], approach, approachControlParam, approachSamplingParam)


# processingSimilarityWithAllDirectoryImages(imFileNames, 'N', 510, 10)
processingSimilarityWithAllDirectoryImages(imFileNames, 'E', 0.3, 10)

# im1, im2 = 0, 1
# #rsp, av = calculateSimilarityScore(imFileNames[im1], imFileNames[im2], 'N', 300, 90)
# rsp, av = calculateSimilarityScore(imFileNames[im1], imFileNames[im2], 'E', 0.3, 10)
# plt.plot((*zip(*rsp)))
# plt.show()


# pts, y = normalCurveFunction(180, 90) # 20 - 1000
# pts, y = expCurveFunction(0.3, 90) # 0.001 - 0.3
# segment space 5 - 90
# plt.plot(*zip(*pts))
# print(y)
# plt.show()


# def indentifySimilarImages(imageList):
#     npImageArrList = []
#     for imageName in imageList:
#         npImageArrList.append(np.array(Image.open(imageName)))



# print(identifyExactDuplicates(imFileNames))

# image = Image.open(imFileNames[0])
# print(image.format, image.size, image.mode)
# image = image.resize(getPercentOfImageSize(image, 20), resample=Image.LANCZOS)
# npimage = np.array(image)
# print(npimage)
# image = image.convert('I')
# image.show()

# imshow(misc.imresize(im.imread(imFileNames[0]), (30, 30), interp='cubic'))
# imshow(np.tile(np.arange(255), (255,1)))