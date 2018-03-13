# Behavioral-Cloning

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Model Architecture and Training Strategy
My final model consisted of the following layers:

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

#### Hyperparameters:

To train the model, I used:

EPOCHS of 5,

BATCH SIZE of 128,

learning rate of 0.004 with Adam optimizer,

MSE loss function,

Train/Validation Split: 0.67/0.33
