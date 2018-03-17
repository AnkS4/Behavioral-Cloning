[//]: # (Image References)

[center]: ./images/center.jpg "Center Camera"
[left]: ./images/left.jpg "Left Camera"
[right]: ./images/right.jpg "Right Camera"
[image]: ./images/image.png "S Channel Image"
[image_f]: ./images/flipped_image.png "Flipped S Image"
[off1]: /images/offtrack1.jpg "Offtrack Image 1"
[off2]: /images/offtrack2.jpg "Offtrack Image 2"

# Behavioral-Cloning

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Model Architecture and Training Strategy
My final model consisted of the following layers:

#### 1. Model Architecture
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Normalized using Lambda         		| 160x320x1 'S' Channel Image from HSV							| 
| Cropping2D      		| Cropped Top 50, Bottom 20 pixels| 
|	Resize using Lambda				|	128x128 Image						|
|			          |												|
| Conv2D  	    | 4x4 stride, SAME padding, Kernel 8x8 	|
| LeakyReLU			|												|
|	Conv2D				|	2x2 stride, SAME padding, Kernel 5x5					|
| LeakyReLU     | 				|
| Conv2D  	    | 2x2 stride, SAME padding, Kernel 5x5	|
| Flatten				|												|
|	Dropout				|	0.2								|
| LeakyReLU     | 				|
| Dense        	|	512							|
|	Dropout       | 0.5     			|
|	LeakyReLU     |												|
|	Dense         | 1					|

#### 2. Hyperparameters:

To train the model, I used:

EPOCHS of 5,

BATCH SIZE of 128,

Optimizer 'Adam',

MSE loss function,

Train/Validation Split: 0.8/0.2

#### 3. Solution Design Approach

My first step was to use a convolution neural network model similar to the model used by comma.ai I thought this model might be appropriate because it is used to also used for Self-Driving Cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set in ratio of 80/20. I previously tried Nvidia End-to-End model & LeNet model, both didn't work for me. I was getting low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I used all three cameras from the dataset.

The training dataset I used is *Udacity dataset*.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like curved lane line. To improve the driving behavior in these cases, I added flip image corresponding to each image.

Then I eventually switched to the different model.

I randomly shuffled the data set and put 33% of the data into a validation set. 

The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by better validation accuracy. I used an adam optimizer so that manually training the learning rate wasn't necessary.

At first, I trained the model using Udacity dataset, it worked fine except getting of the track at two specific spots. So, I retrained the model using *model2.py* using previous output *model.h5*, after training it outputs *model2.h5*.

Spot images from where car was getting off track:

 Offtrack Image 1          |   Offtrack Image 2
:-------------------------:|:-------------------------:
![Offtrack 1][off1]        |  ![Offtrack 2][off2]

For retraining the model I recorded image data by positioning the car when it's ready to go off-road and then steering towards the center.

*video2.mp4* is output produced from *model2.h5*. It is the final output.

At the end of the process, the vehicle was able to drive autonomously for 1 lap.

### Sample Camera Images

Here's the images from 3 differrent cameras at same position:

  Left Camera              |   Center Camera           | Right Camera 
:-------------------------:|:-------------------------:|:-------------------------:
 ![Left Camera][left]      |  ![Center Camera][center] | ![Right Camera][right]
 
 ### Image with it's Flipped Version

I converted the image to HSV using cv2 & only used S (Saturation) channnel from HSV (Hue, Saturation, Value).
As lane lines can easily be identified using S channel.

   S Image                 |   Flipped S Image
:-------------------------:|:-------------------------:
 ![S Image][image]         |  ![Flipped S Image][image_f]
 
 Here, angle for the second image will be negative angle of the first image.
