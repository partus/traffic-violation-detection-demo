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
%config InlineBackend.figure_format = "retina"
plt.rcParams['figure.figsize'] = (18, 9)

from pylsd import lsd

def hough_lines(img,threshold = 80,minLineLength=80,maxLineGap=1):
    #
    rho = 1
    theta = np.pi / 180 / 5
    # threshold is minimum number of intersections in a grid for candidate line to go to output

    # my hough values started closer to the values in the quiz, but got bumped up considerably for the challenge video

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength,
                            maxLineGap)
    return lines

def draw_lines(img, lines, extend=0, color=[255, 0, 0], thickness=2, slip=(0, 0)):
    for line in lines:
        pair = np.array(line[0])
        a = pair[0:2]
        b = pair[2:4]
        a[0] += slip[0]
        a[1] += slip[1]
        b[0] += slip[0]
        b[1] += slip[1]
        vec = a - b
        if extend:
            cv2.line(img, tuple(a + vec * extend),
                     tuple(b - vec * extend), [0, 255, 0], thickness)
        cv2.line(img, tuple(a), tuple(b), color, thickness)
    return img

lsd = cv2.createLineSegmentDetector(0)
def getMainLines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(gray)[0]
    canv = np.zeros(image.shape,dtype=np.uint8)
    draw_lines(canv,lines)
    hlines = hough_lines(canv[...,0],threshold = 80,minLineLength=70,maxLineGap=1)
    return hlines





if __name__ == '__main__':
    imdir = "/data/img/taiwan/"
    # https://stackoverflow.com/questions/41329665/linesegmentdetector-in-opencv-3-with-python
    source_img = os.listdir(imdir)[20]
    source_img = imdir+ source_img
    im= cv2.imread(source_img)
    hlines = getMainLines(im)
    draw_lines(im, hlines)
    plt.imshow(im)
