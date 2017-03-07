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

IMAGE_SIZE = (420, 580)
CROP_SIZE = (120, 120)
CROP_STRIDE = (40, 40)
CROP_DENSITY = 20  # I actually meant stride


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



class PreCompute():
    def __init__(self):
        crops_no_y = (IMAGE_SIZE[0] - CROP_SIZE[0]) / CROP_DENSITY
        crops_no_x = (IMAGE_SIZE[1] - CROP_SIZE[1]) / CROP_DENSITY

        crop_y = CROP_DENSITY * np.arange(crops_no_y) + CROP_SIZE[0] / 2
        crop_x = CROP_DENSITY * np.arange(crops_no_x) + CROP_SIZE[1] / 2

        y = np.arange(IMAGE_SIZE[0], dtype=np.float32)
        x = np.arange(IMAGE_SIZE[1], dtype=np.float32)

        self.argy = np.subtract.outer(crop_y, y)
        self.argx = np.subtract.outer(crop_x, x)

    def get(self):
        return self.argy, self.argx

precomputed_args = PreCompute()
ipdb.set_trace()


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


def store_train_predictions():
    """
    Depends on ordering of the pictures.
    """
    model = get_unet()
    model.load_weights(WEIGHTS_FNAME)

    crops_no_y = (IMAGE_SIZE[0] - CROP_SIZE[0]) / CROP_DENSITY
    crops_no_x = (IMAGE_SIZE[1] - CROP_SIZE[1]) / CROP_DENSITY

    predictions = np.zeros(
        (len(IMAGE_NAMES), crops_no_y, crops_no_x), dtype=np.float32
    )

    for idx, img_name in tqdm.tqdm(enumerate(IMAGE_NAMES)):
        img = load_train_image(img_name)
        for y in xrange(int(crops_no_y)):
            for x in xrange(int(crops_no_x)):
                crop_y = CROP_DENSITY * y
                crop_x = CROP_DENSITY * x

                # print "Crop starts {}, {}".format(crop_y, crop_x)
                crop = make_crop(img, crop_y, crop_x)
                predictions[idx][y][x] = _get_prediction(model, crop)

    np.save(
        os.path.join(PROCESSED_DATA_PATH, 'train_predictions.npy'),
        predictions
    )


if __name__ == "__main__":
    train()
