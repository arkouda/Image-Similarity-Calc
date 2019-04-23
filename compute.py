from PIL import Image 
import numpy as np
import os
from math import pow, sqrt
from hashlib import md5
import matplotlib.pyplot as plt
import multiprocessing
from threading import Thread, Lock
from helper import getPercentOfImageSize, filter_files, deleteFileFromDirectory, generateJsonWithThreshold
from itertools import combinations

MAX_THREADS = 20

def normalCurveFunction(variance , segment):
    points = []
    y_sum = 0
    if variance == 0:
        return [],0
    for x in range(33, 96, 62//segment):
        y = pow(np.exp(1), - pow(x - 50, 2) / (2 * variance) ) /sqrt(2 * np.pi * variance)
        points.append((x,y))
        y_sum += y  
        points.reverse()
    return points, y_sum

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
 
def preProcessImage(image1, image2, additionalResize = False, maxImageSize = 1000):
    lowerBoundSize = (min(image1.size[0], image2.size[0]), min(image1.size[1], image2.size[1]))
    image1 = image1.resize(lowerBoundSize, resample=Image.LANCZOS)
    image2 = image2.resize(lowerBoundSize, resample=Image.LANCZOS)
    if max(image1.size) > maxImageSize and additionalResize:
        resizeFactor = maxImageSize / max(image1.size)
        image1 = image1.resize((int (lowerBoundSize[0] * resizeFactor), int (lowerBoundSize[1] * resizeFactor)), resample=Image.LANCZOS)
        image2 = image2.resize((int (lowerBoundSize[0] * resizeFactor), int (lowerBoundSize[1] * resizeFactor)), resample=Image.LANCZOS)
    return image1, image2

def hammingResize(image1, image2, resizeFactor):
    image1 = image1.resize(getPercentOfImageSize(image1, resizeFactor), resample=Image.BILINEAR)
    image2 = image2.resize(getPercentOfImageSize(image2, resizeFactor), resample=Image.BILINEAR)
    image1 = image1.convert('I')
    image2 = image2.convert('I')
    npImage1 = np.array(image1)
    npImage2 = np.array(image2)
    return npImage1, npImage2

def getSSIMIndex(imageName1, imageName2, windowSize = 7, dynamicRange = 255):
    image1, image2 = Image.open(imageName1), Image.open(imageName2)
    if min(list(image1.size) + list(image2.size)) < 7:
        raise "Image size too small for SSIM approach"
    image1, image2 = preProcessImage(image1, image2, True)
    image1, image2 = image1.convert('I'), image2.convert('I')
    c1 = (dynamicRange * 0.01) ** 2
    c2 = (dynamicRange * 0.03) ** 2
    pixelLength = windowSize ** 2
    ssim = 0.0
    adjustedWidth = image1.size[0] // windowSize * windowSize
    adjustedHeight = image1.size[1] // windowSize * windowSize
    for i in range(0, adjustedWidth, windowSize):
        for j in range(0, adjustedHeight, windowSize):
            cropBox = (i, j, i + windowSize, j + windowSize)
            window1 = image1.crop(cropBox)
            window2 = image2.crop(cropBox)
            npWindow1, npWindow2 = np.array(window1).flatten(), np.array(window2).flatten()
            npVar1, npVar2 = np.var(npWindow1), np.var(npWindow2)
            npAvg1, npAvg2 = np.average(npWindow1), np.average(npWindow2)
            cov = (np.sum(npWindow1 * npWindow2) - (np.sum(npWindow1) * np.sum(window2) / pixelLength)) / pixelLength
            ssim += ((2.0 * npAvg1 * npAvg2 + c1) * (2.0 * cov + c2)) / ((npAvg1 ** 2 + npAvg2 ** 2 + c1) * (npVar1 + npVar2 + c2))
    similarityPercent = (ssim * pixelLength / (adjustedHeight * adjustedWidth)) * 100
    return (imageName1, imageName2, round(similarityPercent, 2))

def getHammingSimilarityIndex(imName1, imName2):
    image1=Image.open(imName1)
    image2=Image.open(imName2)
    origSize = min(image1.size[0], image2.size[0]) * min(image1.size[1], image2.size[1])
    image1, image2 = preProcessImage(image1, image2)
    assert image1.size[0] * image1.size[1] == origSize
    cumulativeSimilarityScore = 0
    samplePts, sampleSum = normalCurveFunction(300, 10)
    for (resizeFactor, factorWeightage) in samplePts:
        npImage1, npImage2 = hammingResize(image1, image2, resizeFactor)
        if (npImage1.size / origSize < 0.1) or (npImage2.size / origSize < 0.1):
            for (x, y) in samplePts:
                if x <= resizeFactor:
                    sampleSum -= y
                    print((x,y), 'delete kar diya', imName1, imName2, npImage1.size, origSize)
                    samplePts.remove((x, y))
        else:
        # if True:
            npGradient1 = np.diff(npImage1) > 1
            npGradient2 = np.diff(npImage2) > 1
            currentSimilarityScore = (np.count_nonzero(np.logical_not(np.logical_xor(npGradient1, npGradient2)))/npGradient1.size)
            weightedSimilarityScore = factorWeightage * currentSimilarityScore
            cumulativeSimilarityScore += weightedSimilarityScore
    averageSimilarityScore = (cumulativeSimilarityScore / sampleSum) * 100
    return (imName1,imName2, round(averageSimilarityScore, 2))

def processImagesWithMultiprocessing(imFileNames, approach, threshold):
    params = list(combinations(imFileNames,2))
    numberOfThreads = min([int(len(params) * 0.1), MAX_THREADS])
    imgTupleWithPercentList = []
    with multiprocessing.Pool(processes=numberOfThreads) as pool:
        if approach == 'N':
            imgTupleWithPercentList = pool.starmap(getHammingSimilarityIndex, params)
        elif approach == 'E':
            imgTupleWithPercentList = pool.starmap(getSSIMIndex, params)
    finalJSON = generateJsonWithThreshold(imgTupleWithPercentList,threshold)
    return finalJSON

def processSimilarity(imFileNames, approach, threshold):
    params = combinations(imFileNames,2)
    imgTupleWithPercentList = []

    if approach == 'N':
        for (im1,im2) in params:
            similarityTuple = getHammingSimilarityIndex(im1, im2)
            imgTupleWithPercentList.append(similarityTuple)
    elif approach == 'E':
        for (im1,im2) in params:
            similarityTuple = getSSIMIndex(im1, im2)
            imgTupleWithPercentList.append(similarityTuple)
    
    finalJSON = generateJsonWithThreshold(imgTupleWithPercentList, threshold)
    return finalJSON

def driverFunction(imFilePATH, approach, threshold, multiprocessingFlag, fileDeleteFlag):
    os.chdir(imFilePATH)
    imFileNames = filter_files(os.listdir())
    print("Files: ", imFileNames)
    finalJSON={}
    if multiprocessingFlag == 'OFF':
        finalJSON = processSimilarity(imFileNames, approach, threshold)
    else:
        finalJSON = processImagesWithMultiprocessing(imFileNames, approach, threshold)
    return finalJSON