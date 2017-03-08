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

from sklearn.preprocessing import StandardScaler


def run_extractors():
    """ Run the extraction function and save the outputs to the shelve """
    last_img = None
    img = None
    diff = None
    diffs = []

    vehicle_img_list = list(glob.glob("data/vehicles/**/*"))
    nonvehicle_img_list = list(glob.glob("data/non-vehicles/**/*"))

    # print "Number of vehicle images = {}".format(len(vehicle_img_list))
    # print "Number of non-vehicle images = {}".format(len(nonvehicle_img_list))
    # print "Balanced!"


    print "Extracting the vehicle features..."
    df = extract_features(vehicle_img_list)

    print "Saving the vehicle features..."
    with open('vehicle-features.p', 'wb') as f:
        pickle.dump(df, f)

    print "Done!"

    print "Extracting the non-vehicle features..."

    print "Saving the non-vehicle features..."
    df = extract_features(nonvehicle_img_list)
    with open('non-vehicle-features.p', 'wb') as f:
        pickle.dump(df, f)

    print "Done!"



def prepare_standard_scaler():
    print "Preparing standard scaler"

    print "Loading the vehicle features..."
    with open('vehicle-features.p', 'rb') as f:
        veh = pickle.load(f)
        y_veh = np.ones(len(veh))

    print "Loading the non-vehicle features..."
    with open('non-vehicle-features.p', 'rb') as f:
        non_veh = pickle.load(f)
        y_non_veh = np.zeros(len(non_veh))

    print "Making one joint dataframe"
    df_x = pd.concat([veh, non_veh], axis=0)

    print "Scaling the features"
    scaler = StandardScaler()
    scaler.fit(df_x.values)
    return scaler


def prepare_data():
    """
    Stuff to do:
        1) Merge data into one dataset, adding ys
        2)
    """


    print "Loading the vehicle features..."
    with open('vehicle-features.p', 'rb') as f:
        veh = pickle.load(f)
        y_veh = np.ones(len(veh))

    print "Loading the non-vehicle features..."
    with open('non-vehicle-features.p', 'rb') as f:
        non_veh = pickle.load(f)
        y_non_veh = np.zeros(len(non_veh))

    print "Making one joint dataframe"
    df_x = pd.concat([veh, non_veh], axis=0)

    print "Scaling the features"
    scaler = StandardScaler()
    scaler.fit(df_x.values)
    print "Fit the scaling transform"
    x_norm = scaler.transform(df_x.values)
    y = np.append(np.ones(len(veh)), np.zeros(len(non_veh)))
    print "Rescaled the features"

    print "Writing the features to the file again..."
    with open('features.p', 'wb') as f:
        pickle.dump(x_norm, f)

    with open('target.p', 'wb') as f:
        pickle.dump(y, f)
    print "Done! Features ready to use."



def imread(fname):
    """
    Wrapper around file readingg

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


def feature_hog(img, transform):
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
    elif transform == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        raise Exception("{} transform is not supported".format(transform))

    return skimage.feature.hog(img)


def feature_resize_flatten(img, out_size=(16, 16)):
    """ Resize to very small size and flatten the output """
    return cv2.resize(img, out_size).ravel()


FEATURE_2_FUNC = {
    'red_hog': lambda img: feature_hog(img, 'red'),
    'blue_hog': lambda img: feature_hog(img, 'blue'),
    'green_hog': lambda img: feature_hog(img, 'green'),
    'gray_hog': lambda img: feature_hog(img, 'gray'),
    'resize_flatten_16x16': lambda img: feature_resize_flatten(img, (16, 16)),
    'resize_flatten_4x4': lambda img: feature_resize_flatten(img, (4, 4))
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
         )
         for key, value in acc.iteritems()
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


if __name__ == "__main__":
    run_extractors()
    # prepare_data()
