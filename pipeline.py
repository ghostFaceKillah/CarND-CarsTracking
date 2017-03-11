import argparse
import ipdb

from sklearn.externals import joblib


def pipeline(next_img, config):
    """
    unpack config 
    """
    pass



if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='Video file.')
    # parser.add_argument('MODEL', help='name of the pickle with model')
    # parser.add_argument('--in', help='input video file')
    # parser.add_argument('--out', help='output video file')
    # args = parser.parse_args()

    model_fname = 'models/model.pkl'
    classifier = joblib.load(model_fname)
