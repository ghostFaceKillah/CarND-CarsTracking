"""
In this file I take a look at the predictions of the model.

TODO:
- write code making predicitons on the test dataset
- implement shear / etc generator manual test
- try a lot more architectures for this setup
"""

import cv2
import itertools as it
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tqdm

from data_crops import (
    CROP_SIZE, IMAGE_SIZE, CROP_DENSITY,
    PROCESSED_DATA_PATH,
    preload_all_train_images,
    preload_all_test_images
)

from utils import plot, plot_color, plot_heatmap, Memoize

# slide the window every CROP_DENSITY pixels

PREDICTION_THRESHOLD = 7.7

# If using the gaussian density
SIGMA = 800.0
RICH_SIGMA_GRID = [
   10., 20., 50., 100.,
   200., 400., 700., 750.,
   800., 850., 900., 1200.,
   1600., 2400., 3200., 5000.
]

SPEC_SIGMA_GRID = list(np.arange(70., 200., 10.))

THRESHOLD_GRID = [0.3, 0.4, .5, .6, .7, .8, .9, .95]

MEM = joblib.Memory(cachedir=PROCESSED_DATA_PATH)


def grays_to_RGB(img):
    """ turn 2D grayscale image into grayscale RGB """
    return np.dstack((img, img, img))


def mark_crop(img, crop_y, crop_x):
    """
    Mark the image crop by drawing a green rectangle around the crop on
    the orignal image.
    """
    img_rgb = grays_to_RGB(img)
    cv2.rectangle(
        img_rgb,
        (crop_x, crop_y),
        (crop_x + CROP_SIZE[1], crop_y + CROP_SIZE[0]),
        (255, 0, 0),
        2
    )
    return img_rgb


def load_stored_predictions(which):
    """
    As predictions are pretty exprensive, we store them on hard drive.
    These are seeded by functions in model_output.py
    """
    if which == 'train':
        name = 'train_predictions.npy'
    elif which == 'test':
        name = 'test_predictions.npy'

    return np.load(os.path.join(PROCESSED_DATA_PATH, name))


def _dice_coeff(y_true, y_hat, smooth=1.0):
    intersection = (y_true * y_hat).sum()
    return (2 * intersection + smooth) / (y_true.sum() + y_hat.sum() + smooth)


def compute_max_of_gaussian_mask(sigma, plot_debug=False):
    crops_no_y = (IMAGE_SIZE[0] - CROP_SIZE[0]) / CROP_DENSITY
    crops_no_x = (IMAGE_SIZE[1] - CROP_SIZE[1]) / CROP_DENSITY

    heat_map = compute_prediciton_heatmap_vectorized(
        np.ones((crops_no_y, crops_no_x)),
        sigma
    )
    if plot_debug:
        plot_heatmap(heat_map, 'gaussian_heatmap_{}.png'.format(sigma))
        print "Max of mask is {}".format(heat_map.max())

    return heat_map.max(), heat_map


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


def compute_prediciton_heatmap_vectorized(predictions, sigma):
    argy, argx = precomputed_args.get()
    yy = np.exp(-0.5 * (1 / sigma) * argy ** 2)
    xx = np.exp(-0.5 * (1 / sigma) * argx ** 2)

    return np.dot(np.dot(yy.T, predictions), xx)


def evaluate_grid_point(static_data, params):
    imgs, masks, predictions = static_data
    sigma, threshold = params
    max_of_heatmap, _ = compute_max_of_gaussian_mask(sigma)

    dice = 0

    for idx in xrange(imgs.shape[0]):
        mask = masks[idx]

        heat_map = compute_prediciton_heatmap_vectorized(predictions[idx], sigma)
        heat_map_norm = heat_map / max_of_heatmap

        binary_heatmap = (heat_map_norm > threshold).astype(np.uint8)
        dice += _dice_coeff(mask, binary_heatmap)

    return dice


@MEM.cache
def optimize_gaussian_mask_params(sigma_grid=RICH_SIGMA_GRID,
                                  threshold_grid=THRESHOLD_GRID):
    """
    Function that outputs the (cutoff, gaussian variance) params values that
    maximize dice coefficient on train.

    From simple experiments it looks as if the gaussian mask is suitable
    for expressing masks. Please check pictures in on_test_gaussian.

    We do the optimization by performing grid search (brute force solution)

    We have to ignore the empty masks it seems...
    """
    predictions = load_stored_predictions('train')
    sigmas, thresholds, dices = [], [], []

    imgs, masks = preload_all_train_images()

    for sigma, threshold in tqdm.tqdm(it.product(sigma_grid, threshold_grid)):
        dice = evaluate_grid_point(
            (imgs, masks, predictions),
            (sigma, threshold)
        )
        sigmas.append(sigma)
        thresholds.append(threshold)
        dices.append(dice)

        print "sigma = {}, threshold = {}, dice = {}".format(
            sigma, threshold, dice
        )

    return sigmas, thresholds, dices


@MEM.cache
def sigma_optimize_gaussian_mask(sigma_grid):
    """
    With the 'window' misconception / bug
    """
    predictions = load_stored_predictions('train')

    imgs, masks = preload_all_train_images()

    mx, window = compute_max_of_gaussian_mask(200.0, False)
    window /= mx

    dices = []

    for sigma in sigma_grid:
        dice = 0
        max_of_heatmap, _ = compute_max_of_gaussian_mask(sigma)

        for idx in tqdm.tqdm(xrange(imgs.shape[0])):
            mask = masks[idx]

            heat_map = compute_prediciton_heatmap_vectorized(
                predictions[idx], sigma
            )
            heat_map_norm = heat_map / max_of_heatmap
            # heat_map_norm *= window (This was misguided - I interpreted
            # a bug in the validate code as a downside of the NN model

            dice += _dice_coeff(mask, heat_map_norm, 0.0)

            dices.append(dice)

        print "sigma = {}, dice = {}".format(sigma, dice)

    return RICH_SIGMA_GRID, dices


@MEM.cache
def show_one_sigma_point(sigma, threshold, plot_details):
    predictions = load_stored_predictions('train')

    imgs, masks = preload_all_train_images()

    max_of_heatmap, _ = compute_max_of_gaussian_mask(sigma, True)

    dices = []

    for idx in tqdm.tqdm(xrange(imgs.shape[0])):
        img = imgs[idx]
        mask = masks[idx]

        heat_map = compute_prediciton_heatmap_vectorized(
            predictions[idx], sigma
        )
        heat_map_norm = heat_map / max_of_heatmap
        binary_heatmap = (heat_map_norm > threshold).astype(np.uint8)

        dices.append(_dice_coeff(mask, binary_heatmap))

        if plot_details:
            plot(img, "img_{}_1_raw.png".format(idx))
            plot_heatmap(heat_map_norm, "img_{}_2_heatmap.png".format(idx))
            plot(binary_heatmap, 'img_{}_3_prediction.png'.format(idx))
            plot(mask, "img_{}_4_target.png".format(idx))

    return pd.DataFrame(pd.Series(dices))


def prepare_2d_optimization_heatmap():
    """
    Present results of 2d optimization (gamma, threshold) results.

    The results are not too nice - they made irrelevant by the fact that
    there are many zero masks.

    The maximum is in the point 3200, 0.3, which is equivalent to sending
    nothing almost always.
    """
    import seaborn
    sigma_grid = [120., 200., 250.]
    thresh_grid = [0.9, 0.91, 0.925, 0.975]
    sigmas, thresholds, dices = optimize_gaussian_mask_params(
        sigma_grid,
        thresh_grid
    )
    df = pd.DataFrame({
        'dice': dices
    }, index=[sigmas, thresholds]).unstack()
    seaborn.heatmap(df)
    plt.savefig('heatmap.png')
    plt.close()


def prepare_1d_optimization_results():

    sigma_grid = SPEC_SIGMA_GRID
    index, dices = sigma_optimize_gaussian_mask(sigma_grid)

    print "www"


def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


def eval_test(sigma, threshold, make_plots):
    predictions = load_stored_predictions('test')

    imgs = preload_all_test_images()

    max_of_heatmap, _ = compute_max_of_gaussian_mask(sigma, False)

    rles = []
    ids = []

    for idx in tqdm.tqdm(xrange(imgs.shape[0])):
        img = imgs[idx]

        heat_map = compute_prediciton_heatmap_vectorized(
            predictions[idx], sigma
        )
        heat_map_norm = heat_map / max_of_heatmap
        binary_heatmap = (heat_map_norm > threshold).astype(np.uint8)

        if make_plots:
            plot(img, "img_{}_1_raw.png".format(idx))
            plot_heatmap(heat_map_norm, "img_{}_2_heatmap.png".format(idx))
            plot(binary_heatmap, 'img_{}_3_prediction.png'.format(idx))

        rle = run_length_enc(binary_heatmap)

        rles.append(rle)
        ids.append(idx + 1)

    first_row = 'img,pixels'
    file_name = 'submission.csv'

    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in xrange(imgs.shape[0]):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')


if __name__ == '__main__':
    # sample_false_positives()
    # store_train_predictions()
    # store_test_predictions()

    # sigma_optimize_gaussian_mask()

    # dices = show_one_sigma_point(120., 0.30, False)
    # print dices.describe().round(2)
    eval_test(200., 0.92, True)

    # prepare_2d_optimization_heatmap()
    # prepare_1d_optimization_results()

