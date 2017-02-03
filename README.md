#**Use Deep Learning to Clone Driving Behavior**

When humans learn to drive, we use our experience to drive the same route with new circumstances and also to drive new routes. In this project we use a simulator to teach a model how we would drive around a track. Afterwards, the model should mimic the human driving behaviour (behavioural cloning) and drive around the same track and a new track - which was not used for training. The agents that runs the model will only control the the steering.



# Model Architecture
The model architecture used for this project is inspired on the paper 'End to End Learning for Self-Driving Cars' by NVIDIA (see resources) and Comma.ai's implementation (see resources). The final model has 7 convolutional layers and 4 fully connected layers. To avoid overfitting multiple max pooling (after 3, 5, and 7 convolution layers) and dropouts have been added. The dropouts are placed after each max pooling layer and each fully connected layer with a value of 0.5. 
Exponential linear units (ELUs) are used as an activation function between the layers. ELUs speed up learning indeep neural networks and avoid a vanishing gradient via the identity for positive values (https://arxiv.org/pdf/1511.07289v1.pdf).

![alt tag](https://github.com/indradenbakker/self-driving-cars-behavioural-cloning/images/model.png?raw=true)


Total params: 1,369,997
Trainable params: 1,369,997
Non-trainable params: 0
