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
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


#used below
def get_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (int(x1), int(y1)), (int(x2),int(y2)), color, thickness)
    return img
def draw_inf_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (int(x1), int(y1)), (int(x2),int(y2)), color, thickness)
    return img


def zero_img(img):
    return np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)


def hough_lines(img):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    #rho and theta are the distance and angular resolution of the grid in Hough space
    #same values as quiz
    rho = 4
    theta = np.pi/180
    #threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 80
    max_line_gap = 70
    #my hough values started closer to the values in the quiz, but got bumped up considerably for the challenge video

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=70,
                            maxLineGap=44)
    return lines


def max_lines(lines):
    def tan(line):
        x1,y1,x2,y2 = line[0]
        return (x1-x2)/(y1-y2)
    tans = list(map(tan,lines))
    sortindexes = np.argsort(tans)
    sigma = 0.5
    delta = 0.3
    tans.sort()
    arctans = np.arctan(tans)
    linecount=len(tans)
    i=0
    numlines=[0]*linecount
    while i<linecount:
        j=i+1
        while j< linecount:
            if arctans[j]< arctans[i]+sigma:
                numlines[i]+=1
                j+=1
            else:
                break
        i+=1
    max1 = numlines.index(max(numlines))
    max1e = max1+numlines[max1]
    print(max1, max1e)
    numlinesseros = list(numlines)
    for i in range(0,linecount):
        if arctans[i] >= arctans[max1]-delta and arctans[i] <= arctans[max1e]+delta:
            numlinesseros[i]=0
    max2 = numlinesseros.index(max(numlinesseros))
    print(numlines)
    print(numlinesseros)
    print(arctans)
    print(max1,max2)
    print(max1,numlines[max1])
    print(max2,numlines[max2])
    print("end")
    maxlines1, maxlines2 = [],[]
    for i in range(max1,max1+numlines[max1]):
        maxlines1.append(lines[sortindexes[i]])
    for i in range(max2,max2+numlines[max2]):
        maxlines2.append(lines[sortindexes[i]])
    return maxlines1,maxlines2


# Python 3 has support for cool math symbols.

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


def canny_image(image):
    gauss_gray=image;
    #same as quiz values
    low_threshold = 50
    high_threshold = 120
    canny_edges = canny(gauss_gray,low_threshold,high_threshold)
    return canny_edges




def slope_intersect(line):
    [[x1, y1, x2, y2]] = line
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b

import itertools

def _keep_finite(line_params):
    print(np.nanpercentile(line_params, 0.9, axis=0))
    print(np.nanpercentile(line_params, 0.1, axis=0))

    tmp = np.array([x for x in line_params if not any(np.isinf(z) for z in x)])
    # plt.hist(tmp[:, 0], bins=1000)
    print(np.percentile(tmp, 95, axis=0))
    up = np.percentile(tmp, 95, axis=0)
    print('up', up)
    low = np.percentile(tmp, 5, axis=0)
    print('tmp.shape', tmp.shape)
    print(tmp[:10])
    print(tmp[:10] < up)
    tmp = tmp[np.all(tmp < up, axis=1)]
    tmp = tmp[np.all(tmp > low, axis=1)]
    plt.hist2d(tmp[:, 0], tmp[:, 1], bins=50)
    # plt.hist(tmp[:, 0], bins=100)
    print(tmp.shape)
    print(np.percentile(tmp, 0.05, axis=0))


for index, source_img in enumerate(sorted(os.listdir("/data/img/taiwan/"))):
    if index < 12:
        continue
    first_frame = 1
    image = mpimg.imread("/data/img/taiwan/"+source_img)
    gaus_gr = gaus_gray(image)
    mpimg.imsave("/data/img/masked/"+source_img,gaus_gr)
    cannyImg = canny_image(gaus_gr)
    # mpimg.imsave("/data/img/canny/"+source_img,cannyImg)
    lines = hough_lines(cannyImg)
    lineImg = draw_lines(zero_img(image), lines)
    plt.imshow(zero_img(image))
    result = weighted_img(lineImg, image)
    h = image.shape[0]
    w = image.shape[1]
    limage = np.zeros((h*5,w*5,result.shape[2]))
    print(h,w,result.shape,limage.shape)
    limage[ h*2:h*3 ,w*2:w*3,:] = result
    plt.imshow(result)
    # break
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
    # print('finite', finite_line_params)
    print(len(finite_line_params))
    pairwise_line_params_of_line_params = []
    for (a1, b1), (a2, b2) in itertools.combinations(finite_line_params, r=2):
        pairwise_line_params_of_line_params.append(slope_intersect([[a1, b1, a2, b2]]))
    # _keep_finite(pairwise_line_params_of_line_params)
    # break
    # plt.hist2d(*zip(*_keep_finite(pairwise_line_params_of_line_params)))
    # break
    aaa, bbb = zip(*finite_line_params)
    plt.hist2d(aaa, bbb, bins=100)
    a_range = np.linspace(min(aaa), max(aaa), 100)
    b_range = -800 * a_range + 10
    plt.plot(a_range, b_range, color='green')

    b2_range = -250 * a_range + 80
    plt.plot(a_range, b2_range, color='blue')

    break
    max1,max2 = max_lines(lines)
    print('max1', len(max1))
    plt.hist2d(*zip(*[slope_intersect(l) for l in max2]))
    plt.xlim(-1, 1)
    plt.ylim(-500, 500)
    break
    max1 = [l for l in max1 if slope_intersect(l)[0] < -.2]
    max2 = [l for l in max2 if slope_intersect(l)[1] > 50]
    draw_lines(lineImg, max1,color=[0,255,0])
    draw_lines(lineImg, max2,color=[0,0,255])
    processed = weighted_img(lineImg, image, α=0.8, β=1., λ=0.)
    mpimg.imsave("taiwan_out/"+source_img,processed)
    plt.clf()
    plt.imshow(processed)
    break
