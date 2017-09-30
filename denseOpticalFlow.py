import cv2
from functions import scaleFrame
import numpy as np
import matplotlib.pyplot as plt


def updateModel(flow):
    global mflow
    mask = flow*flow
    mask = np.sum(mask,axis=2)
    # mask = (flow > 1) | (flow < -1)
    shape = list(mask.shape)
    shape.append(1)
    mask.shape = tuple(shape)
    # mask.reshape(shape)
    # print(mask.shape,shape)
    mask = mask > 1
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
class FlowModel:
    def __init__(self,frame):
        hgh,wdth,chn = frame.shape
        self.mflow = np.zeros((hgh,wdth,2))
    def applyFlow(self, next):
        prvs = self.prvs
        return cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    def updateModel(self,frame):
        next = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        flow = self.applyFlow(next)
        flow = cutLow(flow,0.1)
        #  update
        mask = flow*flow
        mask = np.sum(mask,axis=2)
        # mask = (flow > 1) | (flow < -1)
        shape = list(mask.shape)
        shape.append(1)
        mask.shape = tuple(shape)
        mask = mask > 1
        mflow = self.mflow
        np.add(flow*0.003,mflow*0.997,out=mflow,where=mask)
        self.prvs = next
        return mflow
    def getModel():
        return self.mflow

if __name__ == '__main__':
    # cap = cv2.VideoCapture("/data/livetraffic/2017-07-18/City of Auburn Toomer's Corner Webcam 2-yJAk_FozAmI.mp4")
    # cap = cv2.VideoCapture("/data/livetraffic/2017-08-27/3/tokyo.mp4")
    cap = cv2.VideoCapture("/data/livetraffic/2017-07-18/taiwan.mp4")
    cap.set(cv2.CAP_PROP_FPS, 200)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 102e3)
    ret, frame1 = cap.read()
    model = FlowModel(frame1)
    # frame1 = scaleFrame(frame1)
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    framenum = 0

    while(framenum < 4000):
        ret, frame2 = cap.read()
        framenum+=1
        if not framenum%10:
            print(framenum)

        # frame2 = scaleFrame(frame2)
        if ret:
            cv2.imshow('model',model.getModel())
            # cv2.imshow('flow',flowToImage(cutLow(flow,0.5)))

            # hist = np.histogram(mag,100)

            # cv2.imshow('frame3',hist)
            cv2.imshow('video',frame2)
        k = cv2.waitKey(10) & 0xff

        if k == 27:
            break
        if k==ord('q'):
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',bgr)
    np.save('/data/np/flow_taiwan.npy', mflow)
    # np.save('/data/np/flow_tokyo.npy', mflow)
    cap.release()
    cv2.destroyAllWindows()
