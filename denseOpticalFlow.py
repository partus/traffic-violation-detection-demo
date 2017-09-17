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
def updateModel(flow):
    global mflow
    mask = (flow > 0.1) | (flow < -0.1)
    np.add(flow*0.003,mflow*0.997,out=mflow,where=mask)

def cutLow(ar, threshold):
    mask = (ar > threshold) | (ar < -threshold)
    res=np.zeros(ar.shape)
    np.add(res,ar,out=res,where=mask)
    return res



def flowToImage(flow):
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
while(framenum < 80000):
    ret, frame2 = cap.read()
    if framenum == 10:
        plt.hist(frame2.ravel(),255)
        plt.show
        cv2.imwrite('/data/road.png', frame2)
    framenum+=1
    if not framenum%10:
        print(framenum)

    frame2 = scaleFrame(frame2)
    if ret:
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        if framenum == 10:
            plt.hist(frame2.ravel(),255)
            plt.show
        # if not framenum % 4:
        updateModel(cutLow(flow,0.1))
        cv2.imshow('model',flowToImage(mflow))
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        # im2, contours, hierarchy = cv2.findContours(flow[...,0],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # hull =[]
        # for cont in contours:
        #     hl = cv2.convexHull(cont)
        #     if cv2.contourArea(hl) > 300:
        #         hull.append(hl)
        # fimage = flowToImage(flow)
        # cv2.drawContours(fimage, hull, -1, (0,255,0), 1)
        cv2.imshow('flow',cutLow(mag,0.5))

        # hist = np.histogram(mag,100)
        # if not framenum % 90:
        #     plt.subplot(2,1,1)
        #     plt.hist(np.sign(flow.ravel()*-1),100)
        #     plt.ylim([0,4000])
        #     plt.subplot(2,1,2)
        #     plt.hist(mag.ravel(),100)
        #     plt.ylim([0,4000])
        # plt.show()
        prvs = next

        # cv2.imshow('frame3',hist)
        cv2.imshow('video',cutLow(frame2,150))
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',bgr)
        cv2.imwrite('opticalhsv.png',bgr)
cap.release()
cv2.destroyAllWindows()
