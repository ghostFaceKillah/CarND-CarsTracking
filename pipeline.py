import argparse
import collections
import cv2
import glob
import ipdb
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tqdm

from scipy.ndimage.measurements import label
from sklearn.externals import joblib
from moviepy.editor import VideoFileClip

from windows import (
    make_windows, process_one_image,
    draw_boxes_from_heatmap, heatmap_to_bounding_boxes
)

from features import imread


def pipeline_cached(img, context):
    clf = context['clf']
    tracker = context['tracker']

    current_heatmap = process_one_image(img, context['windows'], clf)

    context['heatmaps'].append(current_heatmap)
    thresh_heatmap = sum(context['heatmaps'])

    thresh_heatmap[thresh_heatmap < context['heatmap_threshold']] = 0
    cv2.GaussianBlur(thresh_heatmap, (31,31), 0, dst=thresh_heatmap)

    bounding_boxes = heatmap_to_bounding_boxes(thresh_heatmap)

    img_labelled = draw_boxes_from_heatmap(np.copy(img), thresh_heatmap)
    # Smooth the predictions
    # tracker.track(bounding_boxes)
    # im2 = tracker.draw_bboxes(np.copy(image))

    return img_labelled


def pipeline_non_cached(img, context):
    clf = context['clf']

    current_heatmap = process_one_image(img, context['windows'], clf)

    thresh_heatmap = current_heatmap
    thresh_heatmap[thresh_heatmap < context['heatmap_threshold']] = 0
    cv2.GaussianBlur(thresh_heatmap, (31,31), 0, dst=thresh_heatmap)

    img_labelled = draw_boxes_from_heatmap(np.copy(img), thresh_heatmap)

    return img_labelled


def initialize_context(context, img_size=(720, 1280)):
    context['windows'] = make_windows(img_size)
    context['heatmaps'] = collections.deque(maxlen=25)
    context['last_heatmap'] = np.zeros(img_size)
    # cache['tracker'] = VehicleTracker(image.shape)


def old_main():
    # parser = argparse.ArgumentParser(description='Video file.')
    # parser.add_argument('MODEL', help='name of the pickle with model')
    # parser.add_argument('--in', help='input video file')
    # parser.add_argument('--out', help='output video file')
    # args = parser.parse_args()

    print 'Loading model ...'
    model_fname = 'models/model.pkl'

    in_file = 'vid/project_video.mp4'
    out_file = 'out/out.mp4'

    context = {}
    initialize_context(context)
    context['clf'] = joblib.load(model_fname)
    context['heatmap_threshold'] = 1
    # context['heatmap_threshold'] = 10


    # Test classification on something easier
    img_list = list(glob.glob('data/test_images/*'))

    # img_name = 'data/test_images/test1.jpg'
    # img_name = 'data/test_images/straight_lines2.jpg'

    for idx, img_name in tqdm.tqdm(zip(range(len(img_list)), img_list)):
      img = imread(img_name)

      out_img = pipeline_non_cached(img, context)

      plt.imshow(out_img)
      fname_suffix = img_name.split('/')[-1].split('.')[0]
      plt.savefig('out/output_{}.png'.format(fname_suffix))
      plt.close()

    # print 'Processing video ...'
    # clip = VideoFileClip(in_file)
    # out_clip = clip.fl_image(lambda i: pipeline_non_cached(i, context))
    # out_clip.write_videofile(out_file, audio=False)


def one():
    model_fname = 'models/model.pkl'


    context = {}
    initialize_context(context)
    context['clf'] = joblib.load(model_fname)
    context['heatmap_threshold'] = 1

    # Test classification on something easier

    img_name = 'data/test_images/test1.jpg'
    # img_name = 'data/test_images/test3.jpg'

    img = imread(img_name)

    out_img = pipeline_non_cached(img, context)

    plt.imshow(out_img)
    fname_suffix = img_name.split('/')[-1].split('.')[0]
    plt.savefig('out/output_{}.png'.format(fname_suffix))
    plt.close()


def many():
    model_fname = 'models/model.pkl'


    context = {}
    initialize_context(context)
    context['clf'] = joblib.load(model_fname)
    context['heatmap_threshold'] = 1

    # Test classification on something easier
    img_list = list(glob.glob('data/test_images/*'))

    # img_name = 'data/test_images/test1.jpg'
    # img_name = 'data/test_images/straight_lines2.jpg'

    for idx, img_name in tqdm.tqdm(zip(range(len(img_list)), img_list)):
        print img_name
        img = imread(img_name)

        out_img = pipeline_non_cached(img, context)

        plt.imshow(out_img)
        fname_suffix = img_name.split('/')[-1].split('.')[0]
        plt.savefig('out/output_{}.png'.format(fname_suffix))
        plt.close()


if __name__ == '__main__':
    many()
    # one()
