## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---


[//]: # (Image References)
[image1]: ./output_images/hog_visualization1.png
[image2]: ./output_images/hog_visualization2.png
[image3]: ./output_images/hog_visualization3.png
[image4]: ./output_images/search_windows_example1.png
[image5]: ./output_images/search_windows_example2.png
[image6]: ./output_images/example_heatmap.png
[image7]: ./output_images/one_frame.png
[video1]: ./output_videos/project_video.mp4

### Feature Extraction and Classifier Training

#### 1. Extract hog features 

In the code cell `[1]` in the ipython notebook, I have implemented the `get_hog_features` using the hog function from `skimage.feature`. This function takes in a lot of parameters for the hog feature derivation. But I mostly experimented with `orientations`, `pixels_per_cell`, and `cells_per_block`, by the following steps:
* I created a `get_random_images` function to just get random images from either `data/vehicles/` or `data/non-vehicles` directories, the search for images in either directory is random and recursive. The user just need to provide a type of image to search, either `car` or `not-car`
* I experimented manually for the following params:
    - `orientations`: 6 ,9, 12
    - `pixels_per_cell`: 4, 8, 16
    - `cells_per_block`: 2, 4, 8, 16
    - `color_spaces`: RGB, YCrCb
* 
![alt text][image1]
![alt text][image2]
![alt text][image3]

#### 2. Deciding on HOG parameters.

As I mentioned on the previous section, I experimented on different HOG parameters, by eye-balling them, it is a little hard to tell exactly which one is better than the other. However, I was able to tell that some combination is definitely worse than the others. Such as when `pixel_per_cell` high as well as `cell_per_block` is high as well. 

But in the next section, I will describe a parameter grid-search pipeline to better find a set of parameters that give good results on the dataset. And it is from that search, I finally decide on the HOG parameters. 

#### 3. Training a classifier

I decided to go with what the course suggested on using a SVM classifier. In code cell `[5]` , I defined a training pipeline to train the SVM. Following these steps:
* get random data from the dataset (or all of them)
* extract HOG features for both `car` and `not-car` types
* extract the spatial and color intensity features from the images as well
* concatenate all these featres and do a normalization on all the inputs using `sklearn`'s `StandardScaler` class to scale everything to unit scale.
* split the dataset into training and testing sets
* use the `sklearn`'s `LinearSVC` class to do the training on the training set
* finally obtain the test score on the test dataset'

#### 4. A hyper-parameter search pipeline

It is hard and subjective to just randomly look at some pictures and decide which combination of hyper-parameter is good. So I decided to implement a pipeline to search for a good combination of hyper-parameter. You can find the code in code cell `[6]`.

Since the training on all dataset takes about 200~300 seconds, it is very expensive to do a comprehensive search, I used some prior intuition from the above section 2 to narrow down the search space. Eventually, I settled down on this set of parameters:
* YCrCb
* 9 orientation
* 8 pixels per cell
* 2 cell per block 
* 32 spatial size
* 32 histogram bin

See Appendix-A for a list of all training outputs. 

### Sliding Window Search

#### 1. First Version of Search

At first, I implemented the `slide_window` and `search_window` functions in code cell `[8]` and `[9]` respectively. `slide_window` is used to generate a list of windows from the image given size and overlapping criteria of the windows. Then `search_window` looks into each one of these window and do a prediction on whether there is a car inside or not. I tried a few combinations of the following parameters:
* window_size: 32, 64, 96
* overlap: 0.25, 0.5
However, I can't find a consistent pattern on which combination is better. Although I could do another parameter search using the grid-search above for the SVC classifier. But I decided to try the scale version of the window search. 

### 2. Second Version of Search
So scale version is one where there is no need to specify overlapping windows, but you specify the number of windows to sample on a given image. Since the number of windows to do a classifier prediction will determine how fast a search go. So having it fixed, it is good that we know how long an overall search would likely take. 
In addition, scale is beneficial in that it can detect cars both far and close. So I decided to use mutiple scales to search. Here are the steps I follow to find the best scales:
* use an initial set of scales (i.e. 1.0 and 1.5) on the entire video. 
* then examine the video and see the area with most problematic false positives
* try a different combination based on what was being detected

#### 3. Examples

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

Again, like I mentioned above, I determined these classifier based on the grid-search pipeline I developed before. 

Here are some example images:

![alt text][image4]
![alt text][image5]
---

### Video Implementation

#### 1. Video Output

My video is very bad, and I have a lot of false positives, I am not sure why this is the case. The SVM training seems to be reasonably good, having achieved 0.99 on the test dataset. 

Here's a [link to my video result](./output_videos/project_video.mp4)


#### 2. Identifying False positives

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are some frames and their corresponding heatmaps:

![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem is that the classifier still produce a lot of false positives despite the use of heatmap, label and thresholding techniques. 

Need to implement vehicle tracking to use history of the detections to inform current detection. 


### Appendix-A 
See file `appendix_A.md` for run results.