import argparse
import collections
import cv2
import ipdb
import itertools
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage.measurements import label
from sklearn.externals import joblib
from moviepy.editor import VideoFileClip

from windows import (
    make_windows, process_one_image,
    draw_boxes_from_labels, heatmap_to_bounding_boxes
)


def pipeline_cached(img, context):
    clf = context['clf']
    tracker = context['tracker']

    current_heatmap = process_one_image(img, context['windows'], clf)

    context['heatmaps'].append(current_heatmap)
    thresh_heatmap = sum(context['heatmaps'])

    thresh_heatmap[thresh_heatmap < context['heatmap_threshold']] = 0
    cv2.GaussianBlur(thresh_heatmap, (31,31), 0, dst=thresh_heatmap)

    bounding_boxes = heatmap_to_bounding_boxes(thresh_heatmap)

    # Smooth the predictions
    tracker.track(bounding_boxes)
    im2 = tracker.draw_bboxes(np.copy(image))

    return im2


def pipeline_non_cached(img, context):
    clf = context['clf']

    current_heatmap = process_one_image(img, context['windows'], clf)

    thresh_heatmap = current_heatmap
    thresh_heatmap[thresh_heatmap < context['heatmap_threshold']] = 0
    cv2.GaussianBlur(thresh_heatmap, (31,31), 0, dst=thresh_heatmap)

    img_labelled = draw_labeled_bboxes(np.copy(img), thresh_heatmap)

    return img_labelled


def initialize_context(context, img_size=(720, 1280)):
    context['windows'] = make_windows(img_size)
    cache['heatmaps'] = collections.deque(maxlen=25)
    cache['last_heatmap'] = np.zeros(image.shape[:2])
    cache['tracker'] = VehicleTracker(image.shape)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Video file.')
    # parser.add_argument('MODEL', help='name of the pickle with model')
    # parser.add_argument('--in', help='input video file')
    # parser.add_argument('--out', help='output video file')
    # args = parser.parse_args()

    print 'Loading model ...'
    model_fname = 'models/model.pkl'

    in_file = 'test_video.mp4'
    out_file = 'out.mp4'

    context = {}
    context['clf'] = joblib.load(model_fname)
    context['heatmap_threshold'] = 1
    # context['heatmap_threshold'] = 10

    print 'Processing video ...'
    clip = VideoFileClip(in_file)
    out_clip = clip.fl_image(lambda i: pipeline_non_cached(i, context))
    out_clip.write_videofile(out_file, audio=False)

