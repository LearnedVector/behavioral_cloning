**Behavioral Cloning Project**
[//]: # (Image References)

[image1]: ./images/center.jpg "center"
[image2]: ./images/right.jpg "right"
[image3]: ./images/left.jpg "left"
[image4]: ./images/crop.jpg "crop"
[image5]: ./images/behavioral_cloning.gf "gif"

![alt text][image5]

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



---
#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filters. 

The model includes RELU layers to introduce nonlinearity

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the center lane camera, as well as the left and right camera angles.I chose to use the Udacity data provided. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Lenet... I thought this model might be appropriate because it was an easy model to try out and I wanted to see how I can build on top of that model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had similar error rates on both training and validation set. 

The car was still not staying on the road so I implemented the right and left camera angle. I then added a correction value for the steering angle. I started small, around 0.01 and worked my way up until I started gettingt better results. 

After tweaking the model a bit. I decided to go with NVidia's model. It was a big upgrade from previouos model, but the car was still going off the road on the dirt road turns. 

I decided to use Udacity's provided data after a few attempt of collecting my own data. The Udacities data worked better then the one's I've collected. 

I tweaked the correction value for the right and left camera until the car finally made it around the track. 

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

| Layer         		      |     Description	        					                 | 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 65x320x3  image   							             | 
| Convolution 5x5     	 | 2x2 stride, Valid padding, outputs 31x159x24 	  |
| RELU					             |											                                   	|
| Convolution 5x5	      | 2x2 stride, Valid padding, outputs 14x78x36 	|
| RELU					             |												            |
| Convolution 5x5	      | 2x2 stride, Valid padding, outputs 6x38x48 	|
| RELU					             |												            |
| Convolution 3x3	      | 2x2 stride, Valid padding, outputs 3x19x64 	|
| RELU					             |												            |
| Convolution 3x3	      | 2x2 stride, Valid padding, outputs 1x9x64 	|
| RELU					             |												            |
| Flatten					             |												            |
|	Fully Connected              	|	100											|
|	Fully Connected              	|	50											|
|	Fully Connected              	|	10											|
|	Fully Connected              	|	1											|


#### 3. Creation of the Training Set & Training Process

To capture good training data I first recorded a single lap. My model did not do so well so I decided to record 5 laps around the track. My model did a lot better but still was not perfect. I then decided to try Udacity's provided data and that worked best. 

Example of the data collected

![alt text][image4]

I decided to use the left and right camera angles and provide a correction value. This helped my model stay away from the side lines. 

![alt text][image2]
![alt text][image3]

I then cropped each image to only provide data of the road. This helped the model get rid of the noise. 

![alt text][image1]

An Interesting finding is that I did not normalize my data set and I still received the satisfactory results. I did have to use a larger correction steering angle for the left and right camera images for it to work. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the model training loss or validation loss was barely changing. I used an adam optimizer so that manually training the learning rate wasn't necessary.
