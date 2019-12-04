# Object detection with python.
# From: https://towardsdatascience.com/object-detection-with-less-than-10-lines-of-code-using-python-2d28eebc5b11

import os
import subprocess
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import tensorflow
import cvlib

# Download an image to work with.
url = 'https://cdn.pixabay.com/photo/2014/02/01/17/28/apple-256261_1280.jpg'
filename = os.path.basename(url)

if os.path.exists(filename) is not True:
    subprocess.run(["wget", url])

# Load the image.
im = cv2.imread(filename)

# Detect common objects.
bbox, label, conf = cv.detect_common_objects(im)

# Draw these on our image.
output_image = cvlib.object_detection.draw_bbox(im, bbox, label, conf)

plt.imshow(output_image)
plt.savefig("output.png", bbox_inches='tight')

