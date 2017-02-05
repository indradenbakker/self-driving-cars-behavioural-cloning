#**Deep learning to mimic human driving behaviour**

When humans learn to drive, we use our experience to drive the same route with new circumstances and also to drive new routes. In this project we use a simulator to teach a model how we would drive around a track. Afterwards, the model should mimic the human driving behaviour (behavioural cloning) and drive around the same track and a new track - which was not used for training. The agents that runs the model will only control the the steering.



## Model Architecture
The model architecture used for this project is inspired on the paper 'End to End Learning for Self-Driving Cars' by NVIDIA (see resources) and Comma.ai's implementation (see resources). The final model has 7 convolutional layers and 4 fully connected layers. To avoid overfitting multiple max pooling (after 3, 5, and 7 convolution layers) and dropouts have been added. The dropouts are placed after each max pooling layer and each fully connected layer with a value of 0.5. 
Exponential linear units (ELUs) are used as an activation function between the layers. ELUs speed up learning indeep neural networks and avoid a vanishing gradient via the identity for positive values (https://arxiv.org/pdf/1511.07289v1.pdf).

![alt tag](https://github.com/indradenbakker/self-driving-cars-behavioural-cloning/blob/master/images/model.png?raw=true)

Total params: 1,369,997<br>
Trainable params: 1,369,997<br>
Non-trainable params: 0

## Training data
###### Udacity
To train our model, we've used the trainig data provided by Udacity. The training data include images from the left, center, and right camera and the steering angle. Note: while testing only the center camera will be used as input.
In total Udacity's dataset contain 8036 frames, if we include the left and right cameras the total number of training images is 24,108.

![alt tag](https://github.com/indradenbakker/self-driving-cars-behavioural-cloning/blob/master/images/original_images.png?raw=true)

The image size are 160x320x3. First we remove the top 40 pixels and the bottom 20 pixels from the images. The top 40 pixels include the sky and some background, which should not be relevant for the steering angle. The bottom 20 pixels include the front of the car. 

![alt tag](https://github.com/indradenbakker/self-driving-cars-behavioural-cloning/blob/master/images/cropped_image.png?raw=true)

To train our model faster we have resized the images to 64x64x3. We apply these same preprocessing steps before reading images for autonomous simulation. 

![alt tag](https://github.com/indradenbakker/self-driving-cars-behavioural-cloning/blob/master/images/resized_image.png?raw=true)

###### Left and right cameras
It's important to "learn" our model what to do if the car moves away from the center of the road (even for a perfectly trained model on Track 1 this is advisable, because it will make sure the model is able to generalise to other tracks as well). There are two solutions to do this:

1. Add recovery data.
2. Use the left and right camers with an adjusted steering angle. 

We have chosen for option 2, because this is more appropiate for real-world scenarios. This method has also been used in the suggested paper of NVIDIA. Our final model uses an offset of 0.25 to account for the side cameras.

![alt tag](https://github.com/indradenbakker/self-driving-cars-behavioural-cloning/blob/master/images/adjusted_steering.png?raw=true)

###### Transformation and augmentation
The make our model robust we've added transformations and augmentations on our images. We do this randomly in the batch generator so that we don't have to keep all transformed and augmented images in memory.

__Flip image__

Track 1 contains mostly left turns and only one right turn. To remove the bias from our model, we randomly flip every image and adjust the steering angle accordingly.

![alt tag](https://github.com/indradenbakker/self-driving-cars-behavioural-cloning/blob/master/images/flipped_image.png?raw=true)


__Adjust brightness__

To account for differences on the track and future tracks, we've randomly changed the brightness of our training data. This helped a lot for our model to drive around on Track 2 as well. 

![alt tag](https://github.com/indradenbakker/self-driving-cars-behavioural-cloning/blob/master/images/augmented_images.png?raw=true)



## Building our model
First, we started with with the suggested NVIDIA model and build on top of that by testing the progress on each step. All the parameters have been emperically tested. And we tried many of the suggested improvements on Slack by fellow students (thanks!). We've finetuned al parameters until the model was able to drive around multiple laps without crossing any lines. 

## Running the model
The model has been tested on Track 1 and Track 2 with screen resolution 640x480 and fastest graphics quality. The trained model is able to drive around both tracks (with custom throttle: slower for bigger steering angles), for Track 1 it is able to drive around without crossing any lane lines or bumping into objects. For Track 2 the trained model is able to drive until the end of the track without bumping into objects most of the times.
To run the simulator use: `python drive.py model.json`.

## Resources
* End to End Learning for Self-Driving Cars (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
* Comma.ai Research: Steering angle (https://github.com/commaai/research/blob/master/train_steering_model.py)
* Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs) (https://arxiv.org/pdf/1511.07289v1.pdf)
