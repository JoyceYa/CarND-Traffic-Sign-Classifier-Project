# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./ReferImg/Visualization.png "Visualization"
[image2]: ./ReferImg/Preprocessing.png "Preprocessing"
[image3]: ./ReferImg/Original_VS_augmented.png "Original_VS_augmented"
[image4]: ./ReferImg/NewTestImg.png "NewTestImg"
[image5]: ./ReferImg/NewTestResults.png "NewTestResults"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/JoyceYa/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_v2-936.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training images with different labels distribute. 
As we can see from the chart, different classes of traffic signs are not evenly distributed, for example, class 0,which represents "Speed limit (20km/h)"， has far less images in the training set compared with class 1,which represents "Speed limit (30km/h)" ,this might make it more difficult for the classifier to learn the features of "Speed limit (20km/h)" traffic signs, and thus decrease the classifier's generalization ability to classify this category.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


As a first step, I decided to convert the images to grayscale. I tried the color image first ,but it doesn't work well. The reason might be that the traffic signs in the data set can be mostly distinguished by their geometries, besides, the noises might be weakened during the grayscale processing.

As a second step, I sharpened the images to make the geometries in the training images more clear. I used a 9x9 kernel with max value of 2.2.

As a third step, I cropped the images by a margin of 3,because the given training images often have relatively big margins, which might affect the extraction of the features.

As a last step, I normalized the image data because this will help the optimizer to get the best solutions, which eventurally will help to build a better classifier. 

Here's a image after each step of the preprocessing.

![alt text][image2]


Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is that, images in the augmented data set are grayscaled, with geometries more contrasting , which helps to improve the feature learning performance of our classifier during the training process.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 24x24x16 	|
| RELU					|												|
| Dropout				| 0.8 keep_prob									|
| Max pooling	      	| 2x2 stride,  outputs 12x12x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 8x8x30 	|
| RELU					|												|
| Dropout				| 0.8 keep_prob									|
| Max pooling	      	| 2x2 stride,  outputs 4x4x30   				|
| Fully connected		| Input 480,   outputs 120						|
| Fully connected		| Input 120,   outputs 84						|
| Fully connected		| Input 84,   outputs 43						|
| Softmax				|			 									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer, the batch size is 128, the number of epochs is 55, the learning rate is 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 94.1%
* test set accuracy of 93.4%

The first architecture that was tried contains two convolutional layers,without no dropout, its validation set accuracy was less than 90%.

I first made some changes in the data pretreatment process，as well as the learning rate and number of epochs, but it didn't help a lot, the accuracy was still under 93%. Considering the covnet might not learn enough features with only two convolutional layers, another convolutional layer was added. I tuned the preprocessing kernels, and applied dropout functions for regularization, because the training set had high accuracy while the validation set had low accuracy, which indicates over fitting. Finally the accuracy reached 94%.

During the finetuning process, I found the following results:
* Data preprocessing had great influence on the validation accuracy
* If the kernel for sharpening got bigger than a certain value, the training images would start losing useful information ,which would decrease the accuracy.
* Dropout helped to improve the accuracy a lot when the training set had high accuracy while the validation set had low accuracy. In this model, setting the keep probability less than 0.8 would decrease the accuracy to some extend.
* Lower learning rate had the potential to get better accuracy in the end.
* Increasing the number of epochs helped to improve the accuracy to some extend, but would take more time for training.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 12 German traffic signs that I found on the web:

![alt text][image4] 

The "Roundabout mandatory" image might be difficult to classify, because there are less than 250 training images for this class, while most of other classes have more than 1000 images for training.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1-60 km/h       		| Ahead only   									| 
| 2-No entry       		| No entry   									| 
| 3-Stop Sign      		| Stop sign   									| 
| 4-Dangerous curve to the right | Dangerous curve to the right  				|
| 5-Keep right			| Keep right									|
| 6-Turn right ahead	| Turn right ahead				 				|
| 7-No vehicles	   		| Keep right					 				|
| 8-Roundabout mandatory| No entry  					 				|
| 9-General caution		| General caution				 				|
| 10-Pedestrians		| Pedestrians       							|
| 11-Go straight or left| Go straight or left  							|
| 12-Ahead only			| Ahead only        							|


The model was able to correctly guess 9 of the 12 traffic signs, which gives an accuracy of 75%. This is lower than the accuracy on the test set of 93.4%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

For the first image, the model is not sure that this is a 60 km/h sign (probability of 0.52), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .53         			| Ahead only   									| 
| .43     				| 60 km/h 										|
| .04					| Turn left ahead								|
| .00	      			| Go straight or right			 				|
| .00				    | Go straight or left  							|


For the seventh image, the model is sure that this is a Keep right sign (probability of 1.0),but actually this is a No vehicles sign.

For the eighth image, the model is not sure that this is a no entry sign (probability of 0.55), but actually this is a roundabout sign.  The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .55         			| no entry   									| 
| .25     				| roundabout									|
| .19					| Ahead only								|
| .01	      			| Turn left ahead			 				|
| .00				    | Go straight or left  							|

For the rest of the images, the model is sure that these are what the signs really are (probability of 1.0).

Here is an exploratory visualization of the test results.

![alt text][image5] 
