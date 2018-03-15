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
|	Resize using Lambda				|	64x64 Image						|
|			|												|
| Conv2D  	| 4x4 stride, SAME padding, Kernel 8x8 	|
| ELU					|												|
|	Conv2D				|	2x2 stride, SAME padding, Kernel 5x5					|
| ELU	  | 				|
| Conv2D  	| 2x2 stride, SAME padding, Kernel 5x5	|
| Flatten				|												|
|	Dropout				|	0.2								|
| ELU	  | 				|
| Dense     	|	512							|
|	Dropout  | 0.5     			|
|	ELU			    |												|
|	Dense | 1					|

#### 2. Hyperparameters:

To train the model, I used:

EPOCHS of 5,

BATCH SIZE of 128,

learning rate of 0.004 with Adam optimizer,

MSE loss function,

Train/Validation Split: 0.67/0.33

#### 3. Solution Design Approach

My first step was to use a convolution neural network model similar to the model used by comma.ai I thought this model might be appropriate because it is used to also used for Self-Driving Cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set in ratio of 67/33. I previously tried Nvidia End-to-End model & LeNet model, both didn't work for me. I was getting low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I used all three cameras from the dataset.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like curved lane line. To improve the driving behavior in these cases, I added flip image corresponding to each image.

Then I eventually switched to the different model.

I randomly shuffled the data set and put 33% of the data into a validation set. 

The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by better validation accuracy. I used an adam optimizer with 0.004 as learning rate.

At the end of the process, the vehicle was able to drive autonomously for 1 lap.
