import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import math
def filterMax(ar,maximum):
    mask = ar < maximum
    return ar*mask
def filterMaxSat(img,maximum):
    mask = img[...,1] < maximum
    shape = list(mask.shape)
    shape.append(1)
    mask=mask.reshape(shape)
    return img*mask
image = mpimg.imread("/data/img/taiwan/image-001.jpg")
image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
image[...,1]=0
image[...,2]=0
plt.imshow(filterMax(image[...,1],30),cmap="gray")
image[...,1]=filterMax(image[...,1],30)
image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
plt.hist(image[...,1].ravel())
image=filterMaxSat(image,14)
plt.imshow(image)
plt.imshow(image[...,0],cmap="gray")
plt.imshow(image[...,1])
plt.imshow(image[...,2],cmap="gray")
