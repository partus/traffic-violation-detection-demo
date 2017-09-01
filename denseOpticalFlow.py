import cv2
from functions import scaleFrame
import numpy as np
import matplotlib.pyplot as plt

# cap = cv2.VideoCapture("/data/livetraffic/2017-07-18/City of Auburn Toomer's Corner Webcam 2-yJAk_FozAmI.mp4")
cap = cv2.VideoCapture("/data/livetraffic/2017-08-27/3/tokyo.mp4")
cap.set(cv2.CAP_PROP_FPS, 200)
# cap.set(cv2.CAP_PROP_POS_FRAMES, 102e3)
ret, frame1 = cap.read()
frame1 = scaleFrame(frame1)
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
framenum = 0
hgh,wdth,chn = frame1.shape
meanang = np.zeros((hgh,wdth))
meanmag = np.zeros((hgh,wdth))
# flow = np.random.randn(hgh,wdth,2)
# flow = np.zeros((hgh,wdth,2))
# flow+=5
meanmag+=50
meanang+=1
mflow = np.zeros((hgh,wdth,2))



while(framenum < 3000):
    ret, frame2 = cap.read()
    framenum+=1
    if not framenum%10:
        print(framenum)

    frame2 = scaleFrame(frame2)
    if ret:
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mflow = flow*0.003+mflow*0.997
        flow = mflow
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        # it = np.nditer([None,None,mag,ang,meanmag,meanang])
        # for mout, aout,m,a,mm,ma in it:
        #     mout = mm*0.99+m*0.01
            # aout = ma*0.99+a*0.01
        # meanmag = it.operands[0]
        # meanang = it.operands[1]
        # print(mag.shape)
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        # hist = np.histogram(mag)
        # if not framenum % 100:
        #     plt.hist(mag.ravel(),100)
        #     plt.ylim([0,4000])
        plt.show()
        prvs = next
        cv2.imshow('frame2',bgr)
        # cv2.imshow('frame3',hist)
        cv2.imshow('video',frame2)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',bgr)
cap.release()
cv2.destroyAllWindows()
