import json
import cv2
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from os import listdir
from os.path import isfile, isdir, join


DB_PATH = './db/'

winSize = (250,250)
cellSize = (25,25)
blockSize = (50,50)
blockStride = (50,50)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 32
useSignedGradients = True
 
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)

def hog_worker(data_gray):
        ppc = 16
        hog_features = []
        for image in data_gray:
            fd = hog.compute(image)
            hog_features.append(fd)

        return hog_features

def img_to_hogcsv():
    """ Creates a csv for image db
    """
    meme_classes = [c for c in listdir(DB_PATH) if isdir(DB_PATH + c)]
    path_files = {}
    data = {}
    db = {}
    list_hog_images = []

    for i, c in enumerate(meme_classes):
        print(c)
        p = DB_PATH + c + '/'
        path_files[c] = [p + m for m in listdir(p) if isfile(p + m)]
        data[c] = [cv2.resize(cv2.imread(i, 0), (250, 250)) for i in path_files[c]]

        cv2.imshow('image', data[c][0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        list_hog_images.append(np.array(hog_worker(data[c])))
        print(list_hog_images[i].shape)

    for i in range(0, list_hog_images[i].shape[1]):
        feature_name = 'f'
        feature_name += str(i)
        db[feature_name] = []
        db['class'] = []
        db['path'] = []

    for c_i, cat in enumerate(list_hog_images):
        for img in cat:
            for j in range(0, len(img)):
                feature_name = 'f'
                feature_name += str(j)
                db[feature_name].append(img[i][0])

            c = meme_classes[c_i]
            db['class'].append(c)
            db['path'].append(path_files[c])

    db = pd.DataFrame.from_dict(db)
    db.to_csv('./db/hog_features.csv')
    return db

def main():

    db = img_to_hogcsv() #transform img database into hog database
    clf = SVC(gamma='auto')
    #clf = MLPClassifier(hidden_layer_sizes=10)
    # clf = GaussianNB()
    X_keys = list(db.keys())
    X_keys.remove('class')
    X_keys.remove('path')

    X, y = db[X_keys], db['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy: %s" %(str(accuracy_score(y_test, y_pred))))
    print('\n')
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()