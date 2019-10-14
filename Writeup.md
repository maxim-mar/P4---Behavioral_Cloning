# **Behavioral Cloning**

## Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* Writeup.md  summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the convolutional neural network designed by the NVidia team. and described in following paper: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

My model consists of 4 x convolutional layer & 4 fully connected layers, similar to the Nvidia approach. Prior passing the data to the model, the data were normalized using keras lambda layer. After normalizing I've cropped the 50 pixels in the top and 30 in the bottom of the image in order to ensure that the model is not disturb by the surroundings shows in the captured image. The model also includes RELU activation layers to introduce nonlinearity. Complete model is shown below (`2. Final Model Architecture`).

#### 2. Attempts to reduce overfitting in the model

In the first trail run with my model, I haven't used any overfitting protection. As I've compared the accuracy of the test data and of the validation data, I've noticed that the accuracy of the validation data is much worse compared to the test data. This observation shows that the model is just overfitting. In order to avoid overfitting I've implemented the the dropout on two positions of my neural network `line 119` and `line 138`.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 160).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (2 rounds), recovering from the left and right sides of the road (2round) and driving in the opposite direction (1 round).

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to map raw pixels from a front-facing camera directly to steering commands. The system automatically learns internal representations of the necessary processing steps such as detecting useful road features with only the human steering angle as the training signal. The system was never explicitly trained it to detect, for example, the outline of roads

As described, my first step was to use a convolution neural network model similar to the one published by Nvidia team. I thought this model might be appropriate because it has already shown a very good performance while driving the self driving car in the real world, using the data recorded during the driving session.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that I've introduced a dropout function on 2 places within my model, after the 3rd convolutional layer and also after the first fully-connected layer. I've chosen the droput of 50% as starting point. With this value I was able to reach directly a good accuracy performance of my training and validation data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. There are some areas were the vehicle is not driving smoothly over the straight parts of the roads, what is mainly caused by my bad driving skills during the recording of the training data.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

|Layer (type)               |  Output Shape           |   Param #  |
|---|---|---|
lambda_1 (Lambda)          |  (None, 160, 320, 3)  |     0         
cropping2d_1 (Cropping2D) |  (None, 80, 320, 3) |       0         
conv2d_1 (Conv2D)            |(None, 38, 158, 24)    |   1824      
activation_1 (Activation)    |(None, 38, 158, 24)    |   0         
conv2d_2 (Conv2D)            |(None, 17, 77, 36)      |  21636     
activation_2 (Activation)    |(None, 17, 77, 36)      |  0         
max_pooling2d_1 (MaxPooling2 |(None, 8, 38, 36)       |  0         
batch_normalization_1  |(None, 8, 38, 36)       |  144       
conv2d_3 (Conv2D)           | (None, 6, 36, 48)       |  15600     
activation_3 (Activation)  |  (None, 6, 36, 48)       |  0         
dropout_1 (Dropout)        |  (None, 6, 36, 48)       |  0         
batch_normalization_2  |(None, 6, 36, 48)       |  192       
conv2d_4 (Conv2D)          |  (None, 4, 34, 64)       |  27712     
activation_4 (Activation)  | (None, 4, 34, 64)       |  0         
batch_normalization_3  |(None, 4, 34, 64)        |  256       
flatten_1 (Flatten)          |(None, 8704)       |        0         
dense_1 (Dense)             | (None, 1164)              10132620  
activation_5 (Activation)    |(None, 1164)       |        0         
dropout_2 (Dropout)          |(None, 1164)        |       0         
batch_normalization_4 (Batch |(None, 1164)         |      4656      
dense_2 (Dense)             | (None, 100)   |            116500    
activation_6 (Activation)   | (None, 100)  |             0         
batch_normalization_5 (Batch| (None, 100)  |             400       
dense_3 (Dense)              |(None, 50)    |            5050      
activation_7 (Activation)    |(None, 50)       |         0         
batch_normalization_6 (Batch |(None, 50)  |              200       
dense_4 (Dense)             | (None, 1)    |             51        
==============================| ===========================| ========
Total params: |  10,326,841
Trainable params: |  10,323,917
Non-trainable params: |  2,924

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane.I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover in case of drifting off the road. These images show what a recovery looks like following:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Afterwards I have driven the car in the opposite direction in order to create data from a "new" track and to have the model better trained for new situations.

Since the dataset has a lot of images with the car turning left than right (because there are more left turns in the track), I've flipped the image horizontally to simulate turning right and also reverse the corresponding steering angle.

![alt text][image6]
![alt text][image7]

I have randomly shuffled the data set and put 20% of the data into a validation set.

After the collection process, I had XXXXXXXX number of data points. I then preprocessed this data with the normalization and cropping operation as described above, in order to improve the accuracy of the model.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I've used 3 x epochs as it was recommended in the documentation provided during the course. I used an adam optimizer so that manually training the learning rate wasn't necessary.
