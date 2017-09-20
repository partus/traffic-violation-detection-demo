from darknet import Detector
detect = Detector(thresh=0.4,cfg="/data/weights/cfg/yolo.cfg",weights="/data/weights/yolo.weights",metafile="/data/weights/cfg/coco.data")
# print(detect("data/dog.jpg"))
import numpy as np
import cv2
import matplotlib.pylab as plt
# from functions import scaleFrame
def scaleFrame(frame,factor=0.25):
    height,width, layers = frame.shape
    # print(width, height,layers)
    return cv2.resize(frame, (int(width*factor),int(height*factor)))
def drawDetection(objs,frame):
    if objs:
        for obj in objs:
            rect = obj[2]
            pt1 = (int(rect[0]-rect[2]/2),int(rect[1]-rect[3]/2))
            pt2 = (int(rect[0]+rect[2]/2),int(rect[1]+rect[3]/2))
            # print(rect)
            cv2.putText(frame,str(obj[0]),pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),1,cv2.LINE_AA)
            cv2.putText(frame,str(obj[1]),pt2, cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, pt1,pt2, (0,255,0), thickness=1, lineType=8, shift=0)
def detsYoloToSortInput(objs):
    res = []
    for obj in objs:
        rect = obj[2]
        res.append([ int(rect[0]-rect[2]/2), int(rect[1]-rect[3]/2), int(rect[0]+rect[2]/2),int(rect[1]+rect[3]/2)])
    return np.array(res)
colours = np.random.rand(32,3)*255

def drawSortDetections(trackers,frame):
    for d in trackers:
        d = d.astype(np.int32)
        cv2.rectangle(frame,(d[0],d[1]),(d[2],d[3]), colours[d[4]%32,:], thickness=1, lineType=8, shift=0)
def drawSortHistory(history, frame):
    for h in history:
        trck = h[0]
        color = colours[h[1]%32,:]
        for rect in trck:
            d = rect[0].astype(np.int32)
            cv2.rectangle(frame,(d[0],d[1]),(d[2],d[3]), color, thickness=1, lineType=8, shift=0)

cap = cv2.VideoCapture("/data/livetraffic/2017-08-27/3/tokyo.mp4")
# cap = cv2.VideoCapture("/data/livetraffic/2017-07-18/City of Auburn Toomer's Corner Webcam 2-yJAk_FozAmI.mp4")
# cap = cv2.VideoCapture("/data/Simran Official Trailer _ Kangana Ranaut _  Hansal Mehta _ T-Series-_LUe4r6eeQA.mkv")
# cap = cv2.VideoCapture("/data/test1.mkv")
cap.set(cv2.CAP_PROP_POS_FRAMES, 80000)
# fgbg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16,detectShadows=True )
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()
# fgbg = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=200,useHistory=200,isParallel=True)
framenum=0
objs = False
from sort import Sort
motTracker = Sort(max_age=10,min_hits=2)
while(1):
    ret, frame = cap.read()
    framenum+=1
    if ret:
        frame = scaleFrame(frame,factor=0.5)
        cv2.imwrite("/tmp/todetect.jpg",frame)
        # if not framenum % 10:
        # if not framenum% 5:
        objs = detect("/tmp/todetect.jpg")
        dets = detsYoloToSortInput(objs)
        trackers,hist =  motTracker.update(dets,True )
        drawSortDetections(trackers, frame)
        drawSortHistory(hist, frame)
        # drawDetection(objs, frame)
        # binarymask = fgmask > 10
        # ret,thresh = cv2.threshold(fgmask,127,255,0)
        # if framenum > 100:
        #     plt.hist(thresh.ravel(),64)
        #     plt.show()
        # print(fgmask.shape)
        # im2, contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # print (len(contours),hierarchy)
        # contimg = np.zeros(frame.shape)
        # contimg[...,0] = fgmask


        print(framenum)
        cv2.imshow('frame',frame)
    else:
        print("noit ok")
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
