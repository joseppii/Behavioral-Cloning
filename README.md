# **Behavioral Cloning Project Report**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./figure_1.png "MSE graph"
[video1]: ./video.mp4 "Video"
---

## 1 . Files Submitted & Code Quality

The project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 a modelcontaining a trained convolution neural network
* model2.h5 the same model using keras 2.2.1 that uses a GPU
* model.png image of the implemented DNN
* README.md summarizing the results

## 2. Data Collection

Using the Udacity provided simulator a drove the track twice, in each direction and repeated the same procedure for both tracks.

## 3. Model code

The file `model.py` contains the code for training and saving the convolution neural network. The file shows the pipeline I used for data augmentation, training and validating the model, and it contains comments to explain how the code works.

## 4.Model Architecture and Training Strategy

### 4.1 Model architecture

Initially, I started by implementing the Lenet architecure used in the class, but this was found to be inadequate for producing a model suitable for driving around the track. I then implemented the model published by NVIDIAs autonomous vehicule team. I used the method plot from `keras.utils.visualize_util` to produce a visualization of the network architecture (line 97 in `model.py`). The model can be seen in the figure below:

![alt text][image1]

The model consists of a normalization layer followed by a cropping layer, followed by five convolutional layers, followed by four fully connected layers. The data is normalized in the model using a Keras lambda layer (code line 60) and cropped using keras `cropping2D` layer (code line 61).

### 4.2 Overfitting reduction

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 92-93). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track, indeffinatelly.

### 4.3 Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 72). Instead the default values, as set by keras was used. The model mean squared error loss was plotted (code lines 18/26) to be used a mean to visuallize the process and detect overfitting. A batch size of 128 was used and the model trainned for 2 epochs, as signs of overfitting were observed. The graph can be seen in the figure below:

![alt text][image2]

The video demonstrating the performance of the model on the first track can be seen here:

[![Behaviorals](https://i.ytimg.com/vi/TFs49KkSWXo/default.jpg)](https://youtu.be/TFs49KkSWXo)

### 4.4 Training data appropriation

Using the Udacity provided simulator the initial data collection strategy was to drive around each track twice making sure the wheels never leave the track. At first, the data used for trainning consisted from the images taken only from the central camera, but this proved insufficient for trainning the model. The next step was to include the data from the other two cameras adjusting the steering angle for each image accordingly. The steering angle was adjusted by subtracting/adding 0.20 from the original steering angle provided. This data was also proved to be insufficient for trainning the model. Then, I complemented the dataset by driving each track in the opposite direction. For trainning the first model, I used trainning data only from the first track. This data proved to be sufficient for trainning the model and therefore no other data augmentation method was used, as the trainned model performed extrimely well. At this point of the collection process, I had 24739 of data points. I preprocessed all data by normalizing it and cropped a total of 80 pixes off each image. For the images collected from the second stage I tried cropping 60px due to the fact that the second track was not flat and the road had a high inclination extending to a larger area within each captured image. Fnally I randomly shuffled the data set and put 20% of the data into a validation set. I also used a generator to load data into the memory on demand rather than loading the entire dataset in memory.

## Future enchancements

Despite that the model performed exceptionally well on the first track, it did not do so for the second track. This was due to a number of factors. The driving data quality was not equally good as in the case of the first track, due to the difficulty of the stage. The driving trajectory was not smooth and the model as a result was continously correcting the steering angle in a similar way. The angle used for the side cameras, was constant and despite that it worked well for the first track, it should be replaced by a proper algorithm that predicts the correct angle. The version of keras that comes as default (1.2.1) with python does not use the GPU for trainning, hence the process was particularly lengthy. Despite manually recompiling tensorflow for my system, so that keras 2.0+ can be installed (that utilizes the GPU),  speed improvements were marginal. This suggests that a thorough investigation of the model has to be performed in order to detect bottlenecks and improve the execution times. But that would require extensive experimentation, Currenlty, the model needed 12hours for data collected for six runs of the second stage (40.000 images approximatelly), on an 8th gen cpu. This prohibited us from tuning the model's hyper parameters to achieve better performance. The AWS servers used were significantly slower.