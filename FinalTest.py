########################################
# Final file that reads in an image of an equation
# and outputs the solved equation
########################################


import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#from predictNumber import predictNumber
from PN import predictNumber
from DrawBoxes import getBoxes
from solveEquation import solveEquation

#load image
img = cv2.imread('9-2_v1.png')

# box and output numbers
num1, op, num2 = getBoxes(img)

# Note that here, if we didn't have a downloaded CNN model, we would add code that
# calls CNN.py to make our CNN and then use it here. predictNumber() would then be modified
# to load that model rather than the downloaded one

# output final equation, solved
one,two,answer = solveEquation(num1,op,num2)

# Print answer
print(one,op,two,"=",answer)

