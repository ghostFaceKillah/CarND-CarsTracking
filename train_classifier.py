"""
Extract machine learning features from images and save them in an easily
accesible format. Script and module at the same time!
"""

import cv2
import ipdb 
import glob
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import skimage.feature


from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

def train():

    print "Loading the features..."
    with open('features.p', 'rb') as f:
        X = pickle.load(f)

    print "Loading the targets..."
    with open('target.p', 'rb') as f:
        y = pickle.load(f)


    print "Splitting the data into training and testing..."

    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print "Training the classifier..."

    svm = LinearSVC()
    svm.fit(X_train, y_train)

    print "Evalutaing the classifier ..."
    print "{:.2%}".format(svm.score(X_test, y_test))
    print "Done!"


if __name__ == "__main__":
    train()
