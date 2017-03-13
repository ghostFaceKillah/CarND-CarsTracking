Vehicle Detection and Tracking


CRITERIA - Histogram of Oriented Gradients (HOG)

* Explain how (and identify where in your code) you extracted HOG features from
  the training images. Explain how you settled on your final choice of HOG
  parameters.

* give explaination for methods used to extract HOG features, including which
  color space was chosen, which HOG parameters (orientations, pixels_per_cell,
  cells_per_block), and why.

* Describe how (and identify where in your code) you trained a classifier using
  your selected HOG features (and color features if you used them).

   The HOG features extracted from the training data have been used to train a
   classifier, could be SVM, Decision Tree or other. Features should be scaled to
   zero mean and unit variance before training the classifier.


CRITERIA - Sliding Window Search

* Describe how (and identify where in your code) you implemented a sliding window
  search. How did you decide what scales to search and how much to overlap
  windows?

* A sliding window approach has been implemented, where overlapping tiles in each
  test image are classified as vehicle or non-vehicle. Some justification has
  been given for the particular implementation chosen.

* Show some examples of test images to demonstrate how your pipeline is working.

* How did you optimize the performance of your classifier?

* Some discussion is given around how you improved the reliability of the
  classifier i.e., fewer false positives and more reliable car detections (this
  could be things like choice of feature vector, thresholding the decision
  function, hard negative mining etc.)


CRITERIA - Video Implementation
* Provide a link to your final video output.
  Your pipeline should perform reasonably well on the entire project video
  (somewhat wobbly or unstable bounding boxes are ok as long as you are
  identifying the vehicles most of the time with minimal false positives.)

* The sliding-window search plus classifier has been used to search for and
  identify vehicles in the videos provided.
  Video output has been generated with detected vehicle positions drawn
  (bounding boxes, circles, cubes, etc.) on each frame of video.

* Describe how (and identify where in your code) you implemented some kind of
  filter for false positives and some method for combining overlapping bounding
  boxes.

* A method, such as requiring that a detection be found at or near the same
  position in several subsequent frames, (could be a heat map showing the
  location of repeat detections) is implemented as a means of rejecting false
  positives, and this demonstrably reduces the number of false positives. Same or
  similar method used to draw bounding boxes (or circles, cubes, etc.) around
  high-confidence detections where multiple overlapping detections occur.


CRITERIA - Discussion

* Briefly discuss any problems / issues you faced in your implementation of this
  project. Where will your pipeline likely fail? What could you do to make it
  more robust?

* Discussion includes some consideration of problems/issues faced, what could be
  improved about their algorithm/pipeline, and what hypothetical cases would
  cause their pipeline to fail.

