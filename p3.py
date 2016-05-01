import sys
import pandas as pd
import numpy as np
import pylab as pl
from PIL import Image
import os
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

STANDARD_SIZE = (100, 100)
DATA_DIR = "data/"
TEST_FILE = ""

def main():

    #get the file path from the command prompt
    if len(sys.argv) > 1:
        TEST_FILE = sys.argv[1]
    else:
        print ("error: lease specify a file path")
        exit()

    print ("TRAINING STARTED!")

    print ("pulling images from files...")
    #Store image paths and labels
    images = []
    rawlabels = []
    for subdir, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if (subdir.split('/')[1]) != "test":
                rawlabels.append(subdir.split('/')[1])
                images.append(os.path.join(subdir, file))


    print ("converting images to arrays...")
    #Create a massive data array
    data = []
    labels = []
    counter = 0
    for imagePath in images:
        #print imagePath
        img = []
        try:
            img = imgToArray(imagePath)
            data.append(img)
            labels.append(rawlabels[counter])
        except IOError:
            pass
        counter += 1
    data = np.array(data)

    print ("reducing arrays using randomizedPCA...")
    #randomizedPCA on training set
    #this reduces the huge amount of data points
    pca = RandomizedPCA(n_components=4)
    data = pca.fit_transform(data)

    #generate a 2D plot that shows the groupings
    #generatePlot(data,labels)

    print ("using K-closest neighbors to classify data...")
    #fit the KNeighbors classifier
    knn = KNeighborsClassifier()
    knn.fit(data, labels)

    print ("-----------------------------------")
    print ("TESTING STARTED!")
    #test the image
    print "The test image, "+TEST_FILE+" is a:"
    test = string_to_img(TEST_FILE,pca)
    print classify_image(test,knn)


def imgToArray(filename):
    img = Image.open(filename).convert('RGB')
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

def string_to_img(filepath,pca):
    img = Image.open(filepath).convert('RGB')
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1,-1)
    return pca.transform(img_wide)

def classify_image(data,knn):
    preds = knn.predict(data)
    return preds[0]

def generatePlot(data,labels):
    for d,l in zip(data,labels):
        color=''
        marker = ''
        if l == 'dollar':
            color='green'
            label='dollar'
        elif l == 'hash':
            color='blue'
            label='hash'
        elif l == 'hat':
            color='yellow'
            label='hat'
        elif l == 'heart':
            label='heart'
            color='purple'
        elif l == 'smile':
            color='orange'
            label='smile'
        plt.scatter(d[0],d[1],color=color)

    plt.title('RandomizedPCA in 2 Dimensions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


main()
