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

def draw_lines(img, lines, extend=0, color=[255, 0, 0], thickness=1, slip=(0, 0)):
    for line in lines:
        pair = np.array(line[0]).astype('uint32')
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
    # return img

lsd = cv2.createLineSegmentDetector(0)
def getMainLines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(gray)[0]
    canv = np.zeros(image.shape,dtype=np.uint8)
    draw_lines(canv,lines,thickness=2)
    hlines = hough_lines(canv[...,0],threshold = 80,minLineLength=70,maxLineGap=1)
    return hlines

flow = np.load('/data/np/flow_taiwan.npy')
flow = np.load('/data/np/flow_tokyo.npy')
import numpy.ma as ma

def isFront(flowvec,line):
    x1,y1,x2,y2=line[0]
    lvec = np.array([x1-x2,y1-y2])
    lvecnorm =lvec/np.sqrt(np.sum(lvec*lvec))
    fvecnorm = flowvec/(np.sqrt(np.sum(flowvec*flowvec)))
    # return flowvec*normvec
    ret = np.sqrt(np.square(np.sum(fvecnorm*lvecnorm)))
    return ret

def classify(flow, lines):
    shape = flow.shape
    shape = [shape[0],shape[1]]
    means = []
    scalar =[]
    front = []
    parallel =[]
    flowlines = []
    isfs = []
    for line in lines:
        mask = np.full(shape,255,dtype=np.uint8)
        pair = np.array(line[0])
        a = pair[0:2]
        b = pair[2:4]
        cv2.line(mask, tuple(a), tuple(b), 0, 80)
        cv2.line(mask, tuple(a), tuple(b), 255, 50)
        # plt.imshow(mask)
        # plt.hist(mask.ravel())
        # mask = mask.astype("bool")
        flowvec = np.array([ma.array(flow[...,0], mask=mask).mean(),ma.array(flow[...,1], mask=mask).mean()])
        center = (a+b)/2
        fline = np.concatenate((center,center+flowvec*100))
        flowlines.append([fline])
        isf = isFront(flowvec, line)
        print(isf)
        isfs.append(isf)
        if isf < 0.91:
            # scalar.append(isFront(flowvec, line))
            front.append(line)
        else:
            parallel.append(line)
    return flowlines
    # return scalar
    # return means
    # return isfs
    return parallel, front

def flowToImage(flow):
    s= flow.shape
    hsv = np.zeros((s[0],s[1],3),dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # return hsv
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
plt.imshow(flowToImage(flow))

def dispOpticalFlow( Image,Flow,Divisor=10,scaleArow=5 ):
    "Display image with a visualisation of a flow over the top. A divisor controls the density of the quiver plot."
    PictureShape = np.shape(Image)
    #determine number of quiver points there will be
    Imax = int(PictureShape[0]/Divisor)
    Jmax = int(PictureShape[1]/Divisor)
    #create a blank mask, on which lines will be drawn.
    mask = np.zeros_like(Image)
    for i in range(1, Imax):
      for j in range(1, Jmax):
         X1 = (i)*Divisor
         Y1 = (j)*Divisor
         X2 = int(X1 + Flow[X1,Y1,1]*scaleArow)
         Y2 = int(Y1 + Flow[X1,Y1,0]*scaleArow)
        #  X2 = np.clip(X2, 0, PictureShape[0])
        #  Y2 = np.clip(Y2, 0, PictureShape[1])
         #add all the lines to the mask
         mask = cv2.line(mask, (Y1,X1),(Y2,X2), [0, 255, 0], 1)
    #superpose lines onto image
    img = cv2.add(Image,mask)
    return img
    #print image
    # cv2.imshow(name,img)
    # return []



if __name__ == '__main__':
imdir = "/data/img/taiwan/"
# https://stackoverflow.com/questions/41329665/linesegmentdetector-in-opencv-3-with-python
source_img = os.listdir(imdir)[20]
source_img = imdir+ source_img
source_img = '/data/road.png'
source_img = '/data/img/taiwan/image-014.jpg'

im= cv2.imread(source_img)
hlines = getMainLines(im)
draw_lines(im, hlines)
plt.imshow(im)
plt.imshow(im)
parallel,front = classify(flow,hlines)
draw_lines(im,parallel, color=(0,0,255))
draw_lines(im,front, color=(0,255,0))

flowlines = classify(flow,hlines)
flowlines
plt.hist(classify(flow,hlines),100)
draw_lines(im,flowlines, color=(0,255,255))
plt.imshow(dispOpticaldef dispOpticalFlow( Image,Flow,Divisor=10,scaleArow=5 ):
    "Display image with a visualisation of a flow over the top. A divisor controls the density of the quiver plot."
    PictureShape = np.shape(Image)
    #determine number of quiver points there will be
    Imax = int(PictureShape[0]/Divisor)
    Jmax = int(PictureShape[1]/Divisor)
    #create a blank mask, on which lines will be drawn.
    mask = np.zeros_like(Image)
    for i in range(1, Imax):
      for j in range(1, Jmax):
         X1 = (i)*Divisor
         Y1 = (j)*Divisor
         X2 = int(X1 + Flow[X1,Y1,1]*scaleArow)
         Y2 = int(Y1 + Flow[X1,Y1,0]*scaleArow)
        #  X2 = np.clip(X2, 0, PictureShape[0])
        #  Y2 = np.clip(Y2, 0, PictureShape[1])
         #add all the lines to the mask
         mask = cv2.line(mask, (Y1,X1),(Y2,X2), [0, 255, 0], 1)
    #superpose lines onto image
    img = cv2.add(Image,mask)
    return imgFlow(im,flowdef dispOpticalFlow( Image,Flow,Divisor=10,scaleArow=5 ):
    "Display image with a visualisation of a flow over the top. A divisor controls the density of the quiver plot."
    PictureShape = np.shape(Image)
    #determine number of quiver points there will be
    Imax = int(PictureShape[0]/Divisor)
    Jmax = int(PictureShape[1]/Divisor)
    #create a blank mask, on which lines will be drawn.
    mask = np.zeros_like(Image)
    for i in range(1, Imax):
      for j in range(1, Jmax):
         X1 = (i)*Divisor
         Y1 = (j)*Divisor
         X2 = int(X1 + Flow[X1,Y1,1]*scaleArow)
         Y2 = int(Y1 + Flow[X1,Y1,0]*scaleArow)
        #  X2 = np.clip(X2, 0, PictureShape[0])
        #  Y2 = np.clip(Y2, 0, PictureShape[1])
         #add all the lines to the mask
         mask = cv2.line(mask, (Y1,X1),(Y2,X2), [0, 255, 0], 1)
    #superpose lines onto image
    img = cv2.add(Image,mask)
    return img))
plt.imshow(dispOpticalFdef dispOpticalFlow( Image,Flow,Divisor=10,scaleArow=5 ):
    "Display image with a visualisation of a flow over the top. A divisor controls the density of the quiver plot."
    PictureShape = np.shape(Image)def dispOpticalFlow( Image,Flow,Divisor=10,scaleArow=5 ):
    "Display image with a visualisation of a flow over the top. A divisor controls the density of the quiver plot."
    PictureShape = np.shape(Image)
    #determine number of quiver points there will be
    Imax = int(PictureShape[0]/Divisor)
    Jmax = int(PictureShape[1]/Divisor)
    #create a blank mask, on which lines will be drawn.
    mask = np.zeros_like(Image)
    for i in range(1, Imax):
      for j in range(1, Jmax):
         X1 = (i)*Divisor
         Y1 = (j)*Divisor
         X2 = int(X1 + Flow[X1,Y1,1]*scaleArow)
         Y2 = int(Y1 + Flow[X1,Y1,0]*scaleArow)
        #  X2 = np.clip(X2, 0, PictureShape[0])
        #  Y2 = np.clip(Y2, 0, PictureShape[1])
         #add all the lines to the mask
         mask = cv2.line(mask, (Y1,X1),(Y2,X2), [0, 255, 0], 1)
    #superpose lines onto image
    img = cv2.add(Image,mask)
    return img
    #determine number of quiver points there will be
    Imax = int(PictureShape[0]/Divisor)
    Jmax = int(PictureShape[1]/Divisor)
    #create a blank mask, on which lines will be drawn.
    mask = np.zeros_like(Image)
    for i in range(1, Imax):
      for j in range(1, Jmax):
         X1 = (i)*Divisor
         Y1 = (j)*Divisor
         X2 = int(X1 + Flow[X1,Y1,1]*scaleArow)
         Y2 = int(Y1 + Flow[X1,Y1,0]*scaleArow)
        #  X2 = np.clip(X2, 0, PictureShape[0])
        #  Y2 = np.clip(Y2, 0, PictureShape[1])
         #add all the lines to the mask
         mask = cv2.line(mask, (Y1,X1),(Y2,X2), [0, 255, 0], 1)
    #superpose lines onto image
    img = cv2.add(Image,mask)
    return imglow(im,flow,scaleArow=25,Divisor=30))
