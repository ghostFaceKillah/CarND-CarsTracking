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


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

from extract_features import (
    extract_features_one_image, imread, prepare_standard_scaler
)

#IMAGE_SIZE = (420, 580)
IMAGE_SIZE = (720, 1280)
CROP_SIZE = (64, 64)
CROP_STRIDE = (40, 40)
CROP_DENSITY = 20  # I actually meant stride

crops_no_y = (IMAGE_SIZE[0] - CROP_SIZE[0]) / CROP_DENSITY
crops_no_x = (IMAGE_SIZE[1] - CROP_SIZE[1]) / CROP_DENSITY


def train():

    print "Loading the features..."
    with open('features.p', 'rb') as f:
        X = pickle.load(f)

    print "Loading the targets..."
    with open('target.p', 'rb') as f:
        y = pickle.load(f)


    print "Splitting the data into training and testing..."

    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    print "Training the classifier..."

    classifier = LinearSVC()
    # classifier = RandomForestClassifier(n_estimators=250, n_jobs=-1)
    classifier.fit(X_train, y_train)

    print "Evalutaing the classifier ..."
    print "{:.2%}".format(classifier.score(X_test, y_test))
    print "Done!"

    return classifier



class PreCompute():
    def __init__(self):
        crop_y = CROP_DENSITY * np.arange(crops_no_y) + CROP_SIZE[0] / 2
        crop_x = CROP_DENSITY * np.arange(crops_no_x) + CROP_SIZE[1] / 2

        y = np.arange(IMAGE_SIZE[0], dtype=np.float32)
        x = np.arange(IMAGE_SIZE[1], dtype=np.float32)

        self.argy = np.subtract.outer(crop_y, y)
        self.argx = np.subtract.outer(crop_x, x)

    def get(self):
        return self.argy, self.argx

precomputed_args = PreCompute()


def compute_prediciton_heatmap_vectorized(predictions, sigma):
    argy, argx = precomputed_args.get()
    yy = np.exp(-0.5 * (1 / sigma) * argy ** 2)
    xx = np.exp(-0.5 * (1 / sigma) * argx ** 2)

    return np.dot(np.dot(yy.T, predictions), xx)


def make_crop(img, crop_y, crop_x):
    return img[
        crop_y:(crop_y + CROP_SIZE[0]),
        crop_x:(crop_x + CROP_SIZE[1])
    ]


def store_train_predictions(classifier, scaler):
    """
    Depends on ordering of the pictures.
    """

    dir_name = 'data/test_images/'
    img_list = list(glob.glob('data/test_images/*'))

    predictions = np.zeros(
         (len(img_list), crops_no_y, crops_no_x), dtype=np.float32
    )

    for idx, img_name in tqdm.tqdm(enumerate(img_list)):
        img = imread(img_name)
        for y in xrange(int(crops_no_y)):
            for x in xrange(int(crops_no_x)):
                crop_y = CROP_DENSITY * y
                crop_x = CROP_DENSITY * x

                crop = make_crop(img, crop_y, crop_x)

                features = extract_features_one_image(crop)
                features = features.values.reshape(1, -1)

                scaled_features = scaler.transform(features)
                prediction = classifier.predict(scaled_features)

                predictions[idx][y][x] = prediction[0]
        print predictions[idx].mean()

    print "Saving predictions on the test images..."
    np.save('test_predictions.npy', predictions)
    print "Done!"


def run_classification_pipeline_on_test_images():
    classifier = train()
    scaler = prepare_standard_scaler()
    store_train_predictions(classifier, scaler)


def watch_predictions():

    dir_name = 'data/test_images/'
    img_list = list(glob.glob('data/test_images/*'))

    predictions = np.load('test_predictions.npy')

    for idx, img_name in enumerate(img_list):
        print "Processing {}".format(img_name)
        pred = predictions[idx]
        heatmap = compute_prediciton_heatmap_vectorized(pred, 1000.0)
        plt.imshow(heatmap, cmap=plt.get_cmap('gray'))
        fname_suffix = img_name.split('/')[-1].split('.')[0]
        plt.savefig('out/output_{}.png'.format(fname_suffix))
        plt.close()


if __name__ == "__main__":
    watch_predictions()
