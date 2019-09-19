# Object detection with python.
# From: https://towardsdatascience.com/object-detection-with-less-than-10-lines-of-code-using-python-2d28eebc5b11

import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bboximport

im = cv2.imread('apple-256261_640.jpg')
