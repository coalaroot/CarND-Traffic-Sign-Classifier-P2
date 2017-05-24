**Build a Traffic Sign Recognition Project**

[image1]: ./examples/data_chart.png "data_chart"
[image3]: ./examples/post_normalization.png "post_norm"
[image2]: ./examples/pre_normalization.png "pre_norm"
[image4]: ./new_img/1.jpg "1"
[image5]: ./new_img/2.jpg "2"
[image6]: ./new_img/3.jpg "3"
[image7]: ./new_img/4.jpg "4"
[image8]: ./new_img/5.jpg "5"

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
###Data Set Summary & Exploration

####1. Provide a basic summary of the data set.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the signs are distributed.
The minimum number of sign examples is abuot 180 and maximum is above 2000. After trainig and tunig I end up not augmenting the data because my results were ok. 

![image1][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data.

As a first step, I decided to normalize the images because it helps a lot with results of the CNN
I tried different normalization techniques but I got the best results with
x - x.mean())/x.std()
Here is an example of one traffic sign form every category image before and after normalization.

![alt text][image2]
![alt text][image3]

I also thought about grayscaling but after couple minutes searching on the internet I found out that it's not such a good idea.

####2. Describe what your final model architecture looks like:

My final model consisted of the following layers and is a result of my experimenting with different layers but I started out with LeNet:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|				|
| Droupout					|		keep_prob = 0.5										|
|  MaxPooling					|						2x2 stride & kernel size, outputs 14x14x6					|
| Convolutional 5x5					|		1x1 stride, valid padding,	outputs 10x10x16									|
| RELU					|												|
| Droupout					|		keep_prob = 0.5										|
|  MaxPooling					|						2x2 stride & kernel size, outputs 5x5x16					|
| Flatten	    | output 400      									|
| Fully connected		| output 120 									|
| RELU				|   									|
| Droupout					|		keep_prob = 0.5										|
| Fully connected		| output 43 									| 

####3. Describe how you trained your model.

Hyperparameters are as follows:
* batch_size = 128
* n_epochs = 15
* keep_prob = 0.5
* learning_rate = 0.001
* optimizer = AdamOptimizer

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.962
* test set accuracy of 0.939

At the beginning I used LeNet as suggested but I got about 0.90 accuracy on the validation test.
So I searched a little, added dropout - seemed to work. I also deleted the fully connected layer that went from 120 to 80 because I didn't see it's usage here and the results were pretty much the same.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![image4][image4] ![img5][image5] ![img6][image6] 
![img7][image7] ![image8][image8]

The third image might be difficult to classify because it was not in the training dataset. The rest I cropped using gimp and I don't think the model should have any trouble recognizing them.

####2. Discuss the model's predictions on these new traffic signs


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road      		| Priority road   									| 
| Ahead only  			| Ahead only 										|
| 10 km/h					| Ahead only											|
| Keep right	      		| Keep right					 				|
| Traffic signals			| Traffic signals    							|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. I think it's a good result as it really is a 100% of what it could guess!

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 15th[] cell of the Ipython notebook.
I'm not going to go through all the new images because as you look on the results of the softmax probabilities you will see that the first choice is always 1.0 sure (although there is one 9.91288757e-13 but it's such a small number it doesn't even count). What's interesting is that it got over 1.0 probability (if you sum them up) but I guess it's just pythons precision.
