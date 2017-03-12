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

from windows import make_windows, process_one_image, draw_boxes_from_labels


WINDOWS = None



def pipeline_cached(next_img, context):
    clf = context['clf']

    global WINDOWS
    if 'windows' not in context
        WINDOWS = make_windows(next_image.shape[:2])

    if cache['heatmaps'] is None:
        cache['heatmaps'] = collections.deque(maxlen=params['heatmap_cache_length'])
        cache['last_heatmap'] = np.zeros(image.shape[:2])

    if 'tracker' not in cache:
        cache['tracker'] = VehicleTracker(image.shape)
    frame_ctr = cache['frame_ctr']
    tracker = cache['tracker']
    cache['frame_ctr'] += 1

    current_heatmap = process_one_image(next_img, WINDOWS, clf)

    cache['heatmaps'].append(current_heatmap)
    thresh_heatmap = sum(cache['heatmaps'])

    thresh_heatmap[thresh_heatmap < params['heatmap_threshold']] = 0
    cv2.GaussianBlur(thresh_heatmap, (31,31), 0, dst=thresh_heatmap)
    labels, no_cars = label(thresh_heatmap)
    Z = []
    for car_number in range(1, labels[1]+1):
        nonzeroy, nonzerox = np.where(labels[0] == car_number)
        Z.append((np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)))
    tracker.track(Z)
    im2 = tracker.draw_bboxes(np.copy(image))

    return im2


def pipeline_non_cached(next_img, context):
    clf = context['clf']

    if 'windows' not in context:
        context['windows'] = make_windows(next_img.shape[:2])

    windows = context['windows']

    current_heatmap = process_one_image(next_img, windows, clf)

    thresh_heatmap = current_heatmap
    thresh_heatmap[thresh_heatmap < context['heatmap_threshold']] = 0
    cv2.GaussianBlur(thresh_heatmap, (31,31), 0, dst=thresh_heatmap)

    labels = label(thresh_heatmap)
    im2 = draw_labeled_bboxes(np.copy(next_img), labels)

    return im2


def clear_cache():
    pipeline_cached.cache = {
        'last_heatmap': None,
        'heatmaps': None,
        'frame_ctr': 0
    }


if __name__ == '__main__':
    # clear_cache()

    # parser = argparse.ArgumentParser(description='Video file.')
    # parser.add_argument('MODEL', help='name of the pickle with model')
    # parser.add_argument('--in', help='input video file')
    # parser.add_argument('--out', help='output video file')
    # args = parser.parse_args()

    print 'Loading model ...'
    model_fname = 'models/model.pkl'
    classifier = joblib.load(model_fname)

    in_file = 'test_video.mp4'
    out_file = 'out.mp4'

    clear_cache()
    context = {}
    context['clf'] = classifier
    context['heatmap_cache_length'] = 25
    context['heatmap_threshold'] = 1
    # context['heatmap_threshold'] = 10

    print('Processing video ...')
    clip2 = VideoFileClip(in_file)
    vid_clip = clip2.fl_image(lambda i: pipeline_non_cached(i, context))
    vid_clip.write_videofile(out_file, audio=False)

