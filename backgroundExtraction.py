import numpy as np
import cv2
import matplotlib.pylab as plt
from functions import scaleFrame

# cap = cv2.VideoCapture("/data/livetraffic/2017-08-27/3/tokyo.mp4")
cap = cv2.VideoCapture("/data/livetraffic/2017-07-18/taiwan.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 102e3)
# fgbg = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=100,detectShadows=True )
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()
fgbg = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=30,useHistory=50,isParallel=True)
framenum=0

while(1):
    ret, frame = cap.read()
    framenum+=1
    if not framenum % 25:
        cv2.imwrite("img/"+str(framenum)+".png",frame)
    if ret:
        # frame = scaleFrame(frame, factor=0.25)
        if not framenum%100:
            fgmask = fgbg.apply(frame)
            # binarymask = fgmask > 10
            # ret,thresh = cv2.threshold(fgmask,127,255,0)
            # if framenum > 100:
            #     plt.hist(thresh.ravel(),64)
            #     plt.show()
            # print(fgmask.shape)
            im2, contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # print (len(contours),hierarchy)
            contimg = np.zeros(frame.shape)
            # contimg[...,0] = fgmask

            hull = []
            for cont in contours:
                hl = cv2.convexHull(cont)
                if cv2.contourArea(hl) > 300:
                    hull.append(hl)
            # cv2.drawContours(contimg, contours, -1, (0,255,255), 1)
            cv2.drawContours(contimg, hull, -1, (0,255,0), 1)
            cv2.imshow('mog',frame)
            fg1 = (fgmask > 1)
            cv2.imshow('fgmask',fgmask)
            cv2.imshow('background',fgbg.getBackgroundImage())
            cv2.imshow('frame',contimg)
            print(framenum)


        # if framenum > 100:

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
