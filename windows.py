import itertools
import numpy as np
import cv2


from features import extract_features_one_image
from scipy.ndimage.measurements import label


def make_windows_one_type(img_shape=(720, 1280),
                          y_bounds=None,
                          w_size=(64, 64)):
    """
    Returns a list of coordinates of crops of the bigger image that 
    is input to the pipeline
    """
    if y_bounds is None:
        y_bounds = [0, img_shape[0]]

    x_bounds = 0, img_shape[1]

    # What's the x and y span of the region to be covered with windows
    xspan = x_bounds[1] - x_bounds[0]
    yspan = y_bounds[1] - y_bounds[0]

    # number of pixels per step in x/y
    xy_overlap = (0.5, 0.5)
    nx_pix_per_step = np.int(w_size[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(w_size[1] * (1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1

    xs, ys = np.meshgrid(range(nx_windows), range(ny_windows))

    startx = xs * nx_pix_per_step + x_bounds[0]
    endx = startx + w_size[0]
    starty = ys * ny_pix_per_step + y_bounds[0]
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
            ((64, 64),   [400, 600]),
            ((96, 96),   [400, 600]),
            ((128, 128), [450, 600])
        ]]))


def process_one_image(img, windows, clf, dec_threshold=0.35):
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

        if dec_threshold is None:
            dec = clf.predict(test_features)[0]
            if dec != 0:
                hot_windows.append(window)
        else:
            # dec = clf.decision_function(test_features) # for SVM
            dec = clf.predict_proba(test_features)[0][1]
            if dec > dec_threshold:
                hot_windows.append(window)

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
