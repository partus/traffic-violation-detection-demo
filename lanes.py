#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import math
#%matplotlib inline

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def draw_lines(img, lines,extend=0, color=[255, 0, 0], thickness=2,slip=(0,0)):
    for line in lines:
        pair = np.array(line[0])
        a=pair[0:2]
        b=pair[2:4]
        a[0]+=slip[0]
        a[1]+=slip[1]
        b[0]+=slip[0]
        b[1]+=slip[1]
        vec = a-b;
        if extend:
            cv2.line(img, tuple(a+vec*extend), tuple(b-vec*extend), [0,255,0], thickness)
        cv2.line(img, tuple(a), tuple(b), color, thickness)
    return img


def zero_img(img):
    return np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)


def hough_lines(img):
    #
    rho = 4
    theta = np.pi/180/2
    #threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 80
    max_line_gap = 10
    #my hough values started closer to the values in the quiz, but got bumped up considerably for the challenge video

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=160,
                            maxLineGap=44)
    return lines




def weighted_img(img, initial_img, alpha=0.8, beta=1., gama=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, gama)


def gaus_gray(image):
    gray_image = grayscale(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #hsv = [hue, saturation, value]
    #more accurate range for yellow since it is not strictly black, white, r, g, or b

    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 210, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    #mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)
    mask_yw_image = mask_white
    kernel_size = 3
    gauss_gray = gaussian_blur(mask_yw_image,kernel_size)
    return gauss_gray


def canny_image(image,low_threshold = 50,high_threshold = 200):
    return canny(image,low_threshold,high_threshold)

def slope_intersect(line):
    [[x1, y1, x2, y2]] = line
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b



for index, source_img in enumerate(sorted(os.listdir("/data/img/taiwan/"))):
    if index < 12:
        continue
    first_frame = 1
    image = mpimg.imread("/data/img/taiwan/"+source_img)
    gaus_gr = gaus_gray(image)
    mpimg.imsave("/data/img/masked/"+source_img,gaus_gr)
    cannyImg = canny_image(gaus_gr)
    mpimg.imsave("/data/img/canny/"+source_img,cannyImg)
    lines = hough_lines(cannyImg)
    h = image.shape[0]
    w = image.shape[1]
    enl = 1
    limage = np.zeros((h*(enl*2+1),w*(enl*2+1),image.shape[2]), dtype=np.uint8)
    limage[ h*enl:h*(enl+1) ,w*(enl):w*(enl+1),:] = image
    lineImg = draw_lines(limage, lines,extend=40,slip=(w,h))
    # result = weighted_img(lineImg, image)
    # print(h,w,result.shape,limage.shape)

    # plt.imshow(limage)
    plt.imshow(limage)
    plt.show()
    break
    line_params = []
    for line in lines:
        line_params.append(slope_intersect(line))
    # print(line_params)
    finite_line_params = [(a, b) for a, b in line_params
                          if not np.isinf(a)
                          and not np.isinf(b)
                          and np.abs(a) < 10
                          and np.abs(b) < 1000]
    x=np.linspace(0,1000,2)
    # for a,b in finite_line_params:
    #     plt.plot(x,x*a+b)
    # plt.plot(x,x)
    plt.show()
    break
