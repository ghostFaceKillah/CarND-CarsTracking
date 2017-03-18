Vehicle Detection and Tracking


[//]: # (Image References)

[image_hog_features]: ./img/hog_features.jpg "HOG Features Example"
[image_hist_features]: ./img/hist_features.jpg "Histogram Features Example"
[image_lowres_features]: ./img/lowres_features.jpg "Low-res Features Example"
[image_boxes_visualization]: ./img/boxes_visualization.jpg "Visualization of considered crops"
[image_combining_predictions]: ./img/combining_predictions_2.jpg "Combining predictions"

### Feature extraction
#### Histogram of Oriented Gradients (HOG)

This assignment is based around HOG features (and some additional simple space / color features).
The code I use for extracting these features in `features.py`.

I have experimented with two versions of HOG features extraction: 
  - `sklearn.feature.img`
  - `cv2.HOGDescriptor().compute()` 


For `cv2.HOGDescriptor().compute()` the official docmentations says that 
currently only one (!) set of parameters is supported, so we don't have a lot of choice.
These parameters are 9 orientations, 8 pixels per cell and 2 cells per block.

For `sklearn.feature.img` the default are 9 orientations, 8 pixels per cell and 3 
cells per block.

Thus, the defaults for the both implementations are not too far off. As they give reasonable
performance on our problem I have decided to stick with them.

I have decided to choose the Y (luminance) channel from YUV color space as 
input to the HOG computation. This was decided in the course of experimentation.
The choice was based on classifier performance on the test set and on my 
subjective opinion about visualizations of the computed features.
The luminance channel from YUV looks clear, with good contrast.

![HOG features][image_hog_features]


#### Other features

I have implemented some additional features.

  - rescale the whole image to `16x16` pixels, map it to YUV color space and use
  all `16x16x3` pixels of the resulting image as features
![Low-res features][image_lowres_features]
  - histogram of YUV channels of such rescaled image
![histogram features][image_hist_features]: 

### Training and choosing classifier


After obtaining the training features, we can traing the classifier
that will handle the actual detection of car in the input image crops.

The code used for training the classifier is in the file `train_classifier.py`.

I define a classification pipeline that has two parts:
  - `sklearn.preprocessing.StandardScaler`
  - Random Forest

The StandardScaler makes sure that the features are scaled to
zero mean and unit variance before training the classifier.

The pipeline works similarly when other classifiers
are used, for example I have tested Linear SVMs.

The final classifier has 99.7% performance on the test dataset, 
although this number is biased due to a bad split in the test and training dataset.
There are many similar consecutive car images in the input dataset, which currently
can go to train and test dataset separately, making overfitting less penalized in the 
test accuracy.


### Sliding Window Search

The video processing pipeline slides a window across the input image,
using the pixels under the window as an input to the classifier. The 
windows for which classifier predicts presence of a car are merged
back into bounding boxes of cars.


##### Windows
The code for this part of pipeline can be found is `windows.py`.


I first precompute a list of coordinates of all windows.
The windows come in three varieties:
  - `64 x 64` pixels, sweeping heights between `400` and `600` and whole width
  - `96 x 96` pixels, heights in the range `[400, 600]`
  - `128 x 128` pixels, heights in the range `[400, 600]`

How this windows relate to the input image can be seen on the below 
visualization.

![crops visualization][image_boxes_visualization]

I have decided on this particular scales of window search
and the amount of overlap (50%) based on my previous
experiences with object localization in images. It is an often used
default that results in a pretty good results in this setting.

I iterate over the list, using each of the windows coordinates to 
crop out a part of the picture, rescale it as needed and pass it 
to the later parts of the architecture - computation of the features
and classification.
  
  
##### Merging information from multiple windows

The classifications on multiple windows are mixed using a simple algorithm:

* First, the information from overlapping windows is summed into a heatmap.
We start from an array of zeros of the same shape as the input image. For each 
crop produced by the sliding window search, if classifier returns a 
positive prediction, we add one to value of each pixel on the heatmap
that belongs to currently considered crop. Additionally this heamap is summed over
consecutive 20 videos frames. This corresponds to summing over 0.8 second of 
the input video. The predictions are also smoothed with gaussian blur, the motivation being 
mainly smoothing the temporal component.

* Next, I use `scipy.ndimage.measurements.label` utility on the heatmap that
is output of the previous step. This utility assigns unique labels to continuous
regions of non-zero pixels. 

* Finally, I draw bounding boxes around the labelled pixels. I expand each bounding 
box to the smallest rectangle that contains all of the pixels in the given labelled 
region.

![Combining predictions on crops][image_combining_predictions]

I have chosen this implementation as a reasonable starting point due to
its simplicity. It is easy to observe in action and works ok.



###### Optimizing the performance of classifier in the context of pipeline

In order to improve the performance of the pipeline I have tweaked the 
threshold of positive classification.
Random Forest classifier that I have used actually returns a number in the 
range `[0, 1]`. Typically, it predicts the positive class if above
`0.5` threshold. I have brought this threshold down to `0.35`

To compensate for false positives introduced by this, I require a prediction 
to show up for longer period on the temporal smoothing heatmap.


This procedure, however contains a serious flaw. I have tweaked the performance of
the pipeline to the supplied videos. It can be the case that it actually decreases
the ability generalize outside of the particular videos used in this project.


The final video can be seen on youtube:

[![Project video output](https://img.youtube.com/vi/ZB8m3I_nx7o/0.jpg)](https://youtu.be/ZB8m3I_nx7o)

as well in this repository
[project_four.mp4](./out/project_four.mp4)



### Problems, potential problems and ideas for improvement


When it comes to the shortcomings of the implemented pipeline, there are many interesting
areas for improvement.

* Perhaps the most pressing one is improving treatment of the vehicles on
 the opposite lane. As the current pipeline heavily on averaging predictions
 over consecutive frames, it is almost guaranteed to work bad
 for objects that are incoming straight our direction. This is because
 for a long period of time they will remain in a similar place in the image 
 (when they are still far away) and finaly when they start passing us, they will 
 change their position in the picture rapidly. Because of the smoothing over time,
 this movement will not be properly reflected in the bounding box.

* In relation to the above, more sophisticated combining of the predictions using
spatiotemporal structure, for example using Kalman filter.

* In relation to the above, one could try to improve treatment of cars that are
near each other. Based on the velocity information we could try to discern
separate cars.

* More sophisticated split of the training data to make the classification accuracy 
  numbers on the test dataset more reliable. This will enable engineering stronger 
  classifier, which I believe with some tweaking can be reached. The basic ideas such
  as adding more uncorrelated features, averaging over many classifiers could result
  in an improvement.

* Removing dependence on the features that have been gathered in non-extreme lighting
scenarios. For example, by gathering more data in different conditions.

