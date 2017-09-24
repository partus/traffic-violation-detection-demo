from darknet import Detector
detect = Detector(thresh=0.4,cfg="/data/weights/cfg/yolo.cfg",weights="/data/weights/yolo.weights",metafile="/data/weights/cfg/coco.data")
# print(detect("data/dog.jpg"))
import numpy as np
import cv2
import matplotlib.pylab as plt

import asyncio

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
def historyToPolylines(hist):
    ret = []
    for h in hist:
        trck = h[0]
        cont = []
        for rect in trck:
            cont.append((rect[:,0:2]+rect[:,2:4])/2)
        ret.append(np.array(cont, dtype=np.int32))
    return ret

cap = cv2.VideoCapture("/data/livetraffic/2017-08-27/3/tokyo.mp4")
# cap = cv2.VideoCapture("/data/livetraffic/2017-07-18/City of Auburn Toomer's Corner Webcam 2-yJAk_FozAmI.mp4")
# cap = cv2.VideoCapture("/data/Simran Official Trailer _ Kangana Ranaut _  Hansal Mehta _ T-Series-_LUe4r6eeQA.mkv")
# cap = cv2.VideoCapture("/data/livetraffic/2017-07-18/Jackson Hole Wyoming Town Square - SeeJH.com-psfFJR3vZ78.mp4")
# cap.set(cv2.CAP_PROP_POS_FRAMES, 80000)
# fgbg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16,detectShadows=True )
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()
# fgbg = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=200,useHistory=200,isParallel=True)
framenum=0
objs = False
from sort import Sort
motTracker = Sort(max_age=30,min_hits=4)
async def detectAsync():

    objs = detect("/tmp/todetect.jpg")
    return objs
    # return objs
loop = asyncio.get_event_loop()

from lsd import getClassified
from lsd import draw_lines

async def main():
    framenum = 0
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(None, detect, "/tmp/todetect.jpg")
    parallel,front = getClassified(cv2.imread("/tmp/todetect.jpg"))
    print(future.done())
    initiated = False
    while True:
        await asyncio.sleep(0)
        ret, frame = cap.read()
        framenum+=1
        if ret:
            frame = scaleFrame(frame,factor=0.5)

            # print(future.done(),future.cancelled())
            cv2.imwrite("/tmp/todetect.jpg",frame)
            if(future.done()):
                initiated = True
                print("result")
                objs = future.result()
                dets = detsYoloToSortInput(objs)
                trackers,hist = motTracker.update(dets,True )
                pol = historyToPolylines(hist)
                # print(future.result())

                future.cancel()
                future = loop.run_in_executor(None, detect, "/tmp/todetect.jpg")


            # objs = detect("/tmp/todetect.jpg")
            # dets = detsYoloToSortInput(objs)
            # trackers,hist = motTracker.update(dets,True )
            if initiated:
                drawSortDetections(trackers, frame)
                # drawSortHistory(hist, frame)
                # pol = historyToPolylines(hist)
                cv2.polylines(frame, pol, False, (0,255,0))
                draw_lines(frame,parallel, color=(0,0,255))
                draw_lines(frame,front, color=(0,255,0))
            # print(pol)
            # hull = []
            # for cont in pol:
            #     print("cont")
            #     print(cont)
            #     if(len(cont)):
            #         # hl = cv2.convexHull(cont)
            #         hl = cv2.approxPolyDP(cont,30,False)
            #         hull.append(hl)

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

loop.run_until_complete(main())
# loop.run_in_executor(None, detectAsync)
# asyncio.ensure_future(main())
#
# asyncio.ensure_future(main())
# loop.run_forever()
print("complete")
cap.release()
cv2.destroyAllWindows()
