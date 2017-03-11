"""
Extract machine learning features from images and save them in an easily
accesible format. Script and module at the same time!
"""

import cv2
import glob
import ipdb 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import skimage.feature
import time
import tqdm


# for prediction ????? Need to figure this stuff out, check some examples
# IMAGE_SIZE = (720, 1280)

# for training
IMAGE_SIZE = (64, 64)


def prepare_training_dataset():
    """ Run the extraction function and save the outputs to the shelve """
    start = time.time()

    last_img = None
    img = None
    diff = None
    diffs = []

    vehicle_img_list = list(glob.glob("data/vehicles/**/*"))
    nonvehicle_img_list = list(glob.glob("data/non-vehicles/**/*"))

    print "Number of vehicle images = {}".format(len(vehicle_img_list))
    print "Number of non-vehicle images = {}".format(len(nonvehicle_img_list))
    print "The number of two classes is more-or-less balanced!"

    print "Extracting the vehicle features..."
    veh = extract_features(vehicle_img_list)
    print "Done!"

    print "Extracting the non-vehicle features..."
    non_veh = extract_features(nonvehicle_img_list)
    print "Done! {:.0f} s to extract the data".format(time.time() - start)

    start = time.time()
    print "Merging data into one dataset..."

    df_x = pd.concat([veh, non_veh], axis=0)
    y = np.append(np.ones(len(veh)), np.zeros(len(non_veh)))

    print "Writing the features to file ..."
    with open('features.p', 'wb') as f:
        pickle.dump(df_x, f)

    with open('target.p', 'wb') as f:
        pickle.dump(y, f)
    print "Done! It took {:.0f} s, features ready!".format(time.time() - start)


def imread(fname):
    """
    Wrapper around file reading

    TODO: make some kind of assert making sure that we read stuff in well,
          as different types of image files are read in differently 

    cv2.imread -> BGR, 0-255
    matplotlib .png -> RGB, 0, 1
    matplotlib .jpg -> RGB, 0, 255

    but if you make cv2.cvtColor on image scaled from 0 to 1, it will be scaled 0 -
    255 again
    """
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


HOG_DESCRIPTOR = None
def initialize_hog_transform(img_shape=IMAGE_SIZE,
                             orient=9,
                             pix_per_cell=8,
                             cell_per_block=2,
                             reset=False):
    """
    Taken from internet, but changed quite a bit.
    """
    global HOG_DESCRIPTOR
    if HOG_DESCRIPTOR is None or reset:
        cell_size = (pix_per_cell, pix_per_cell)  # h x w in pixels
        block_size = (cell_per_block, cell_per_block)  # h x w in cells
        nbins = orient  # number of orientation bins
        
        # winSize is the size of the image cropped to a multiple of the cell size
        HOG_DESCRIPTOR = cv2.HOGDescriptor(
            _winSize=(
                img_shape[1] // cell_size[1] * cell_size[1],
                img_shape[0] // cell_size[0] * cell_size[0]
            ),   
            # only 16 x 16 block size is supported for now
            _blockSize=( 
                block_size[1] * cell_size[1],
                block_size[0] * cell_size[0]
            ),
            _blockStride=(cell_size[1], cell_size[0]),
            # only 8 x 8 is supported for now
            _cellSize=(cell_size[1], cell_size[0]),
            # only 9 bins are supported for now
            _nbins=nbins
        )

    return HOG_DESCRIPTOR


def feature_hog(img, transform, method='sklearn'):
    """
    Extract Histogram Of Gradients feature.
    Wraps skimage.feature.hog function.
    """
    if transform == 'red':
        img = img[:, :, 0]
    elif transform == 'green':
        img = img[:, :, 1]
    elif transform == 'blue':
        img = img[:, :, 2]
    elif transform == 'yuv_y':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = img[:, :, 0]
    elif transform == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        raise Exception("{} transform is not supported".format(transform))

    if method == 'sklearn':
        return skimage.feature.hog(img)
    elif method == 'cv2':
        hogger = initialize_hog_transform()
        full_hog_features = hogger.compute(img)
        return full_hog_features[:, 0]
    else:
        raise Exception("Unsupported source of hog transorfm")


def feature_resize_flatten(img, out_size=(16, 16)):
    """ Resize to very small size and flatten the output """
    return cv2.resize(img, out_size).ravel()


def feature_color_map_resize_flatten(img, out_size, channel_no=None, color_map=None):
    if color_map == 'rgb':
        img = img
    elif color_map == 'yuv':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    if channel_no is not None:
        img = img[:, :, channel_no]

    return cv2.resize(img, out_size).ravel()


def feature_color_hist(img, bins_no=16, bin_range=(0, 256), color_map=None):
    if color_map == 'yuv':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    return np.concatenate(
        [np.histogram(img[:, :, col], bins=bins_no, range=bin_range)[0]
         for col in [0, 1, 2]]
    )


FEATURE_2_FUNC = {
    # 'red_hog': lambda img: feature_hog(img, 'red'),
    # 'blue_hog': lambda img: feature_hog(img, 'blue'),
    # 'green_hog': lambda img: feature_hog(img, 'green'),
    # Y from yuv
    'yuv_y_hog': lambda img: feature_hog(img, 'yuv_y', method='cv2'),
    # 'gray_hog': lambda img: feature_hog(img, 'gray'),
    # 'resize_flatten_16x16': lambda img: feature_resize_flatten(img, (16, 16)),
    # 'resize_flatten_4x4': lambda img: feature_resize_flatten(img, (4, 4))
    'yuv_resize_flatten_4x4': (
        lambda img: feature_color_map_resize_flatten(img, (16, 16), None, 'yuv')
    ),
    'yuv_channel_hist': lambda img: feature_color_hist(img, color_map='yuv'),
}


def extract_features(image_list, wanted_features=FEATURE_2_FUNC.keys()):
    """
    Loop over images and extract wanted features.
    """

    acc = {feature: [] for feature in wanted_features}
    fnames = []

    for fname in tqdm.tqdm(image_list):
        img = imread(fname)

        for feat_name in wanted_features:
            feature_realization  = FEATURE_2_FUNC[feat_name](img)
            acc[feat_name].append(feature_realization)

    df = pd.concat([
        pd.DataFrame(
            value,
            columns=['{}_{}'.format(key, ix) for ix in xrange(len(value[0]))]
        ) for key, value in acc.iteritems()
    ], axis=1)

    return df


def extract_features_one_image(img, wanted_features=FEATURE_2_FUNC.keys()):
    acc = {}
    fnames = []

    for feat_name in wanted_features:
        acc[feat_name] = FEATURE_2_FUNC[feat_name](img)

    features = pd.concat([
         pd.Series(
             value, 
             index=['{}_{}'.format(key, ix) for ix in xrange(len(value))]
         ) for key, value in acc.iteritems()
    ])

    return features


def measure_dims(image_list):
    """ Check if all of the supplied img dims are the same.  """
    heights = []
    widths = []
    names = []

    for fname in tqdm.tqdm(image_list):
        img = imread(fname)
        heights.append(img.shape[0])
        widths.append(img.shape[1])
        names.append(fname)

    df = pd.DataFrame({
        'h': heights,
        'w': widths
    }, index=names)

    print "No of unique heights = {}".format(len(set(df['h'])))
    print "No of unique widths = {}".format(len(set(df['w'])))

    print "Typical h, w = {}".format(df.iloc[0]['h'], df.iloc[0]['w'])


if __name__ == "__main__":
    prepare_training_dataset()
