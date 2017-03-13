import ipdb
import itertools
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2

import string
import random

from features import extract_features_one_image
from scipy.ndimage.measurements import label


def make_windows_one_type(img_shape=(720, 1280),
                                    y_start_stop=None,
                                    w_size=(64, 64)):
    """
    Returns a list of crops of the bigger image_to_be_considered
    """
    if y_start_stop is None:
        y_start_stop = [0, img_shape[0]]

    x_start_stop = 0, img_shape[1]

    # What's the x and y span of the region to be covered with windows
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # number of pixels per step in x/y
    xy_overlap=(0.5, 0.5)
    nx_pix_per_step = np.int(w_size[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(w_size[1] * (1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1

    xs, ys = np.meshgrid(range(nx_windows), range(ny_windows))

    startx = xs * nx_pix_per_step + x_start_stop[0]
    endx = startx + w_size[0]
    starty = ys * ny_pix_per_step + y_start_stop[0]
    endy = starty + w_size[1]

    window_list = [
        ((sx, sy), (ex, ey)) for sx, ex, sy, ey in 
        zip(*map(lambda x: x.ravel(), [startx, endx, starty, endy]))
    ]
    
    return window_list


def make_windows(image_size):
    return list(itertools.chain(*[
        make_windows_one_type(image_size, y_lims, w_size)
        for w_size, y_lims in [
            ((64, 64),  [400, 500]),
            ((96, 96),  [400, 500]),
            ((128, 128),[450, 600])
        ]
    ]))


def process_one_image(img, windows, clf, dec_threshold=0.75):
    """
    Run the classifier over crops of the image
    """
    hot_windows = []
    windows = list(windows)

    for window in windows:
        (sx, sy), (ex, ey) = window
        test_img = cv2.resize(img[sy:ey, sx:ex], (64, 64))

        features = extract_features_one_image(test_img)

        test_features = np.array(features).reshape(1, -1) 
        # dec = clf.decision_function(test_features)
        dec = clf.predict(test_features)[0]

        if dec != 0:
            hot_windows.append(window)

        # dec = clf.predict_proba(test_features)[0][1]
        # if dec > 0.2:
        #     hot_windows.append(window)

    return make_heatmap(hot_windows, img.shape)


def make_heatmap(hot_windows, image_shape):
    heatmap = np.zeros((image_shape[0], image_shape[1]), np.uint8)

    for (sx, sy), (ex, ey) in hot_windows:
        sx, ex = np.clip((sx, ex), 0, image_shape[1])
        sy, ey = np.clip((sy, ey), 0, image_shape[0])
        xs, ys = np.meshgrid(xrange(sx, ex), xrange(sy, ey))
        heatmap[ys, xs] += 1

    return heatmap


def draw_boxes_from_heatmap(img, heatmap):
    labels, no_cars = label(heatmap)

    for car_number in xrange(no_cars):
        y, x = (labels == car_number + 1).nonzero()
        box_min, box_max = (x.min(), y.min()), (x.max(), y.max())
        cv2.rectangle(img, box_min, box_max, (0, 255, 0), 6)
    return img


def heatmap_to_bounding_boxes(heatmap):
    labels, no_cars = label(heatmap)
    bboxes = []
    for car_number in xrange(no_cars):
        y, x = np.where(labels == car_number + 1)
        bboxes.append(((x.min(), y.min()), (x.max(), y.max())))
    return bboxes



def non_max_suppression_fast(boxes, overlapThresh):
    """
    This code is taken from

    www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    where it is cited as Malisiewicz et al.
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")
