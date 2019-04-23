from PIL import Image 
import numpy as np
import os
from math import pow, sqrt
from hashlib import md5
import matplotlib.pyplot as plt
import multiprocessing
from threading import Thread, Lock
from helper import getPercentOfImageSize, filter_files, deleteFileFromDirectory

mutex = Lock()
numberOfThreads = 20
imgTupleWithPercentList = []

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
    resizeFactor = 0.3
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

def getSSIMIndex(imageName1, imageName2, windowSize = 7):
    image1, image2 = Image.open(imageName1), Image.open(imageName2)
    image1, image2 = preHammingPrep1(image1, image2)
    image1, image2 = image1.convert('I'), image2.convert('I')
    dynamicRange = 255 #4294967295
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
    similarityPercent = (ssim * pixelLength / (adjustedHeight * adjustedWidth))*100
    return (imageName1, imageName2, round(similarityPercent, 2))

def getHammingSimilarityIndex(imName1, imName2):
    image1=Image.open(imName1)
    image2=Image.open(imName2)
    image1, image2 = preHammingPrep1(image1, image2)
    cumulativeSimilarityScore = 0
    samplePts, sampleSum = normalCurveFunction(600, 10)
    for (resizeFactor, factorWeightage) in samplePts:
        npImage1, npImage2 = preHammingPrep2(image1, image2, resizeFactor)
        npGradient1 = np.diff(npImage1) > 1
        npGradient2 = np.diff(npImage2) > 1
        currentSimilarityScore = (np.count_nonzero(np.logical_not(np.logical_xor(npGradient1, npGradient2)))/npGradient1.size)
        weightedSimilarityScore = factorWeightage * currentSimilarityScore
        cumulativeSimilarityScore += weightedSimilarityScore
    averageSimilarityScore = (cumulativeSimilarityScore / sampleSum) * 100
    return (imName1,imName2, round(averageSimilarityScore, 2))

def processingImagesWithMultiprocessing(imFileNames, approach, threshold, imgTupleWithPercentList):
    imgCount = len(imFileNames)
    if imgCount < 1:
        finalJSON = generatingJsonWithThreshold(imgTupleWithPercentList, threshold)
        return finalJSON
    params = [(imFileNames[0], imFileNames[im2])  for im2 in range(1,imgCount)] 
    with multiprocessing.Pool(processes=numberOfThreads) as pool:
        if approach == 'N':
            similarityTupleList = pool.starmap(getHammingSimilarityIndex, params)
        elif approach == 'E':
            similarityTupleList = pool.starmap(getSSIMIndex, params)
        imgTupleWithPercentList = imgTupleWithPercentList + similarityTupleList
    print(similarityTupleList)
    print(len(similarityTupleList))
    return processingImagesWithMultiprocessing(imFileNames[1:], approach, threshold, imgTupleWithPercentList)

def processingSimilarity(imFileNames, approach, threshold):
    imgCount = len(imFileNames)
    if imgCount < 1:
        finalJSON = generatingJsonWithThreshold(imgTupleWithPercentList, threshold)
        return finalJSON
    if approach == 'N':
        for im2 in range(1,imgCount):
            similarityTuple = getHammingSimilarityIndex(imFileNames[0], imFileNames[im2])
            imgTupleWithPercentList.append(similarityTuple)
    elif approach == 'E':
        for im2 in range(1,imgCount):
            similarityTuple = getSSIMIndex(imFileNames[0], imFileNames[im2])
            imgTupleWithPercentList.append(similarityTuple)

    return processingSimilarity(imFileNames[1:], approach, threshold)

def generatingJsonWithThreshold(imgsWithPercentList, threshold):
    finalJSON = {}
    for obj in imgsWithPercentList:
        im1, im2 , percent = obj
        if int(percent) <= threshold:
            continue
        if im1 in finalJSON:
            finalJSON[im1].append([im2,percent])
        else:
            finalJSON[im1] = []
            finalJSON[im1].append([im2,percent])
        if im2 in finalJSON:
            finalJSON[im2].append([im1,percent])
        else:
            finalJSON[im2] = []
            finalJSON[im2].append([im1,percent])
    return finalJSON

def driverFunction(imFilePATH, approach, threshold, multiprocessingFlag, fileDeleteFlag):
    os.chdir(imFilePATH)
    imFileNames = filter_files(os.listdir())
    print(imFileNames)
    
    finalJSON={}

    if multiprocessingFlag == 'OFF':
        finalJSON = processingSimilarity(imFileNames, approach, threshold)
    else:
        finalJSON = processingImagesWithMultiprocessing(imFileNames, approach, threshold, imgTupleWithPercentList)
        
    return finalJSON
