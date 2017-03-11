import ipdb
import numpy as np


def make_windows(img_shape=(720, 1280),
                 y_start_stop=None,
                 w_size=(64, 64),
                 xy_overlap=(0.5, 0.5)):
    """
    Returns a list of crops of the bigger image_to_be_considered
    """
    if y_start_stop is None:
        y_start_stop = [0, img_shape[0]]

    x_start_stop = 0, img_shape[1]

    # Compute the span of the region to be covered with windows
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
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


def create_windows(image_size):
    return [
        make_windows(image_size, y_lims, w_size)
        for w_size, y_lims in [
            ((64, 64),  [400, 500]),
            ((96, 96),  [400, 500]),
            ((128, 128),[450, 600])
        ]
    ]



if __name__ == "__main__":
    # make_windows()
    print create_windows((720, 1280))
