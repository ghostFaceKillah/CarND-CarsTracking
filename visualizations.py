
import cv2

import matplotlib.pyplot as plt
import pandas as pd
from skimage.feature import hog
import seaborn

from features import imread

seaborn.set_style('dark')


"""
Load an example image from the dataset and show hog featues
"""


def show_hog_car():
    fname = 'data/vehicles/GTI_Far/image0008.png'
    img = imread(fname)
    yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    features, hog_image = hog(yuv_img[:, :, 0], visualise=True)

    fig, ax = plt.subplots(1, 3)
    fig.tight_layout()

    ax[0].imshow(img)
    ax[0].set_title("Input image")
    ax[1].imshow(yuv_img[:, :, 0], cmap='gray')
    ax[1].set_title("Luminance (Y) of YUV")
    ax[2].imshow(hog_image, cmap='gray')
    ax[2].set_title("HOG features")
    plt.savefig("img/hog_features.jpg")
    plt.close()



def show_many_car_hogs():
    fnames = [
        'data/vehicles/GTI_Far/image0008.png',
        'data/vehicles/GTI_MiddleClose/image0008.png',
        'data/vehicles/GTI_Left/image0009.png'
    ]
    imgs = [imread(fname) for fname in fnames]

    yuv_imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2YUV) for img in imgs]

    fig, axes = plt.subplots(len(fnames), 3)
    fig.tight_layout()

    for idx, (img, yuv_img) in enumerate(zip(imgs, yuv_imgs)):
        ax = axes[idx]
        features, hog_image = hog(yuv_img[:, :, 0], visualise=True)

        ax[0].imshow(img)
        ax[0].set_title("Input image")
        ax[1].imshow(yuv_img[:, :, 0], cmap='gray')
        ax[1].set_title("Luminance (Y) of YUV")
        ax[2].imshow(hog_image, cmap='gray')
        ax[2].set_title("HOG features")
    plt.savefig("img/hog_features.jpg")
    plt.close()


def show_histogram_features():
    """
    def feature_color_hist(img, bins_no=16, bin_range=(0, 256), color_map=None):
        return np.concatenate(
            [np.histogram(img[:, :, col], bins=bins_no, range=bin_range)[0]
             for col in [0, 1, 2]]
        )
    """

    fname = 'data/vehicles/GTI_Far/image0008.png'
    img = imread(fname)
    yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    fig, ax = plt.subplots(2, 3)
    fig.tight_layout()

    for idx, channel in enumerate("YUV"):
        ax[0][idx].imshow(yuv_img[:, :, idx], cmap='gray')
        ax[0][idx].set_title(channel)

    for idx in range(3):
        data = pd.Series(yuv_img[:, :, idx].ravel())
        data.hist(bins=16, range=(0, 256), ax=ax[1][idx])

    plt.savefig("img/hist_features.jpg")
    plt.close()


def show_lowres_features():
    fname = 'data/vehicles/GTI_Far/image0008.png'
    img = imread(fname)
    yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    smol = cv2.resize(yuv_img, (16, 16))

    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(img)
    ax[0].set_title("Input image")

    for idx, channel in enumerate("YUV"):
        ax[idx + 1].imshow(smol[:, :, idx], cmap='gray', interpolation='none')
        ax[idx + 1].set_title("Lowres {}".format(channel))

    plt.savefig("img/lowres_features.jpg")


def classifier_visualization():
    from sklearn.externals import joblib

    model_fname = 'models/model.pkl'
    model = joblib.load(model_fname)

    rf = model.steps[1][1]
    # feature importances look pretty bad


def show_the_boxes():
    from windows import make_windows_one_type

    fname = 'img/input/test1.jpg'
    img = imread(fname)
    img_size = (720, 1280)
    assert img.shape[:2] == img_size

    windows = [
        ((64, 64),   [400, 600]),
        ((96, 96),   [400, 600]),
        ((128, 128), [400, 600])
    ]

    colors = [
        (255, 0, 0),
        (255, 255, 0),
        (0, 255, 0)
    ]

    for (w_size, y_lims), color in zip(windows, colors):
        windows = make_windows_one_type(img_size, y_lims, w_size)

        for box_min, box_max in windows:
            cv2.rectangle(img, box_min, box_max, color, 2)


    plt.imshow(img)
    plt.savefig("img/boxes_visualization.jpg")



def classification_pipeline():
    fname = 'img/input/test1.jpg'
    img = imread(fname)
    pass


if __name__ == '__main__':
    # show_hog_car()
    # show_many_car_hogs()
    # show_histogram_features()
    # show_lowres_features()
    # classifier_visualization()
    show_the_boxes()
