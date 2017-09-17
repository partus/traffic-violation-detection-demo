# importing some useful packages
import matplotlib.pyplot as plt
import itertools
import matplotlib.image as mpimg
import pandas as pd
from collections import defaultdict
from collections import Counter
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import math
%matplotlib inline

plt.rcParams['figure.figsize'] = (18, 9)

from pylsd import lsd

imdir = "/data/img/taiwan/"
# https://stackoverflow.com/questions/41329665/linesegmentdetector-in-opencv-3-with-python
source_img = os.listdir(imdir)[20]
source_img = imdir+ source_img

im= cv2.imread(source_img)



src = cv2.imread(source_img, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
lines = lsd.lsd(gray)
for i in xrange(lines.shape[0]):
    pt1 = (int(lines[i, 0]), int(lines[i, 1]))
    pt2 = (int(lines[i, 2]), int(lines[i, 3]))
    width = lines[i, 4]
    cv2.line(src, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))
plt.imshow(src)


img = cv2.imread(source_img,0)

#Create default parametrization LSD
lsd = cv2.createLineSegmentDetector(0)
for line in lines:
    print(line[0])
#Detect lines in the image

lines = lsd.detect(img)[0] #Position 0 of the returned tuple are the detected lines
lines.shape
#Draw detected lines in the image
drawn_img = lsd.drawSegments(img,lines)

#Show image
plt.imshow(drawn_img )
