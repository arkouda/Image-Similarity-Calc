import os

acceptedExt = ['jpg', 'jpeg','png','bmp', "tiff",'JPG']

def filter_files(filenames):
    acceptedFiles = []
    for fname in filenames:
        ext = (fname.split('.'))[-1]
        if ext in acceptedExt: 
            acceptedFiles.append(fname)
    return acceptedFiles

def getPercentOfImageSize(image, percent):
    h, w = image.size
    return (int(h * (percent/100)), int(w * (percent/100)))

def deleteFileFromDirectory(dir,images):
    for im in images:
        os.remove(dir+im)

def generateJsonWithThreshold(imgsWithPercentList, threshold):
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

def filterByThreshold(imgsWithPercentList, threshold):
    filteredList = []
    for obj in imgsWithPercentList:
        im1, im2 , percent = obj
        if int(percent) >= threshold:
            filteredList.append((im1, im2, percent))
    return filteredList