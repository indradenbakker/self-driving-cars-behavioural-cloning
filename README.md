#**Use Deep Learning to Clone Driving Behavior**

When humans learn to drive, we use our experience to drive the same route with new circumstances and also to drive new routes. In this project we use a simulator to teach a model how we would drive around a track. Afterwards, the model should mimic the human driving behaviour (behavioural cloning) and drive around the same track and a new track - which was not used for training. The agents that runs the model will only control the the steering.



# Model Architecture
The model architecture used for this project is inspired on the paper 'End to End Learning for Self-Driving Cars' by NVIDIA (see resources) and Comma.ai's implementation (see resources). The final model has 7 convolutional layers and 4 fully connected layers. To avoid overfitting multiple max pooling (after 3, 5, and 7 convolution layers) and dropouts have been added. The dropouts are placed after each max pooling layer and each fully connected layer with a value of 0.5. 
Exponential linear units (ELUs) are used as an activation function between the layers. ELUs speed up learning indeep neural networks and avoid a vanishing gradient via the identity for positive values (https://arxiv.org/pdf/1511.07289v1.pdf).

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 64, 64, 3)     12          lambda_1[0][0]
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 64, 64, 3)     0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 62, 62, 32)    896         elu_1[0][0]
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 62, 62, 32)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 60, 60, 32)    9248        elu_2[0][0]
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 60, 60, 32)    0           convolution2d_3[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 30, 30, 32)    0           elu_3[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 30, 30, 32)    0           maxpooling2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 28, 28, 64)    18496       dropout_1[0][0]
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 28, 28, 64)    0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 26, 26, 64)    36928       elu_4[0][0]
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 26, 26, 64)    0           convolution2d_5[0][0]
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 13, 13, 64)    0           elu_5[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 13, 13, 64)    0           maxpooling2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 11, 11, 128)   73856       dropout_2[0][0]
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 11, 11, 128)   0           convolution2d_6[0][0]
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 9, 9, 128)     147584      elu_6[0][0]
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 9, 9, 128)     0           convolution2d_7[0][0]
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 4, 4, 128)     0           elu_7[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 4, 4, 128)     0           maxpooling2d_3[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2048)          0           dropout_3[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 512)           1049088     flatten_1[0][0]
____________________________________________________________________________________________________
elu_8 (ELU)                      (None, 512)           0           dense_1[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 512)           0           elu_8[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 64)            32832       dropout_4[0][0]
____________________________________________________________________________________________________
elu_9 (ELU)                      (None, 64)            0           dense_2[0][0]
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 64)            0           elu_9[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 16)            1040        dropout_5[0][0]
____________________________________________________________________________________________________
elu_10 (ELU)                     (None, 16)            0           dense_3[0][0]
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 16)            0           elu_10[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             17          dropout_6[0][0]
====================================================================================================

Total params: 1,369,997
Trainable params: 1,369,997
Non-trainable params: 0



