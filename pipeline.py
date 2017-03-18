import collections
import cv2
import numpy as np

from sklearn.externals import joblib
from moviepy.editor import VideoFileClip

from windows import make_windows, process_one_image, draw_boxes_from_heatmap


def pipeline(img, context):
    clf = context['clf']

    current_heatmap = process_one_image(img, context['windows'], clf)

    context['heatmaps'].append(current_heatmap)
    thresh_heatmap = sum(context['heatmaps'])

    thresh_heatmap[thresh_heatmap < 20] = 0
    blurred_heatmap = cv2.GaussianBlur(thresh_heatmap, (31,31), 0)

    img_labelled = draw_boxes_from_heatmap(np.copy(img), blurred_heatmap)

    return img_labelled


def initialize_context(img_size=(720, 1280)):
    return {
        'windows': make_windows(img_size),
        'heatmaps': collections.deque(maxlen=25),
        'last_heatmap': np.zeros(img_size)
    }


if __name__ == '__main__':

    # in_file = 'vid/short_video.mp4'
    # out_file = 'out/short.mp4'

    in_file = 'vid/project_video.mp4'
    out_file = 'out/main.mp4'

    print 'Loading model ...'
    model_fname = 'models/model.pkl'
    context = initialize_context()
    context['clf'] = joblib.load(model_fname)


    print 'Processing video ...'
    clip = VideoFileClip(in_file)
    out_clip = clip.fl_image(lambda i: pipeline(i, context))
    out_clip.write_videofile(out_file, audio=False)
