###############################
# File from website: 
# https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
#
# 
#
###############################

import tensorflow as tf
# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation
from keras.callbacks import EarlyStopping

def ConvNN():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Format data for keras API; this data-preparation section advised by linked website
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    img_size = (28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    #Use sequential model because based on research that is what seems like is the best for classification CNN
    model = Sequential()

    #Perform initial convolution. Activation just linear. 3x3 conv-> valid padding would reduce size to 26x26
    #adding more filters takes longer, inc acc
    model.add(Conv2D(16, kernel_size=(3,3), padding="same", input_shape=img_size))

    #Perform 2nd conv
    #adding more layers takes longer, inc acc
    model.add(Conv2D(32, (3, 3), padding="same", activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25)) #dropout for regularization; .25 kind of arbitrary as rate of disassoc.
    model.add(Conv2D(64, (3, 3), padding="same", activation = "relu"))
    model.add(Dropout(0.25))

    #flatten array for fully connected layers 
    model.add(Flatten())

    #do fc layer with 256 nodes; would do 14x14x64 but that had estimated training time of a day
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    #adding another fully conn doesnt help and makes training super long
    model.add(Dense(10,activation="softmax"))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train,y=y_train, epochs=10)

    model.evaluate(x_test, y_test)

    return model