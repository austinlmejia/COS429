#####################################
# Predicts number based on input image 
# using neural network from keras model
#
#
#####################################

import cv2
import matplotlib.pyplot as plt
import numpy as np
from CNN import ConvNN

def predictNumber(img):

    # trains a CNN on the MNIST data set-> dont know how to import the model without training the CNN
    # each time??
    model = ConvNN()

    # format image for square sizing
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    squareGray = imgGray[1000:3500, 0:2500] #square size depends on img but worked for all sample images
    resizedSquare = cv2.resize(squareGray, (28, 28), interpolation=cv2.INTER_AREA)

    #format for keras
    squareAsArray = resizedSquare.reshape(28, 28, 1)
    squareAsArray = squareAsArray.astype('float32')

    #make black number with white background
    squareAsArray[squareAsArray <= 180.] = 255.
    squareAsArray[squareAsArray < 255.] = 0.

    #format for keras (again)
    squareAsArray /= 255
    predictionImage = squareAsArray.reshape(1, 28, 28, 1)
    
    #visualize image if needed:
    #plt.imshow(predictionImage.reshape(28,28), cmap = "Greys")

    pred = model.predict(predictionImage)
    
    return pred.argmax()