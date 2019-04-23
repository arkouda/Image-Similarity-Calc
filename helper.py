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
