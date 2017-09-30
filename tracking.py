from darknet import Detector

# print(detect("data/dog.jpg"))
import numpy as np
import cv2
import matplotlib.pylab as plt
from sort import Sort
import asyncio
from linetools import seg_poliline_intersect, draw_lines
from lsd import getClassified
from  backgroundExtraction import BackgroundExtractor
from functions import scaleFrame
from denseOpticalFlow import FlowModel
from collections import deque

colours = np.random.rand(32,3)*255

detect = Detector(thresh=0.4,cfg="/data/weights/cfg/yolo.cfg",weights="/data/weights/yolo.weights",metafile="/data/weights/cfg/coco.data")

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

def historyToTracks(hist):
    ret = []
    for h in hist:
        trck = h[0]
        cont = []
        for rect in trck:
            cont.append((rect[:,0:2]+rect[:,2:4])/2)
        ret.append((np.array(cont, dtype=np.int32),h[1]))
    return ret


# cap = cv2.VideoCapture("/data/livetraffic/2017-07-18/City of Auburn Toomer's Corner Webcam 2-yJAk_FozAmI.mp4")
# cap = cv2.VideoCapture("/data/Simran Official Trailer _ Kangana Ranaut _  Hansal Mehta _ T-Series-_LUe4r6eeQA.mkv")
# cap = cv2.VideoCapture("/data/livetraffic/2017-07-18/Jackson Hole Wyoming Town Square - SeeJH.com-psfFJR3vZ78.mp4")
# cap.set(cv2.CAP_PROP_POS_FRAMES, 80000)

motTracker = Sort(max_age=30,min_hits=4)
async def detectAsync():

    objs = detect("/tmp/todetect.jpg")
    return objs
    # return objs
loop = asyncio.get_event_loop()

# async def futurise():

class Executor:
    def __init__(self,func,*args):
        self.func = func
        loop = asyncio.get_event_loop()
        self.future = loop.run_in_executor(None, detect, *args)
    def res(self):
        if self.future.done():
            True


async def main():
    cap = cv2.VideoCapture("/data/livetraffic/2017-08-27/3/tokyo.mp4")
    r0,f0 = cap.read()
    f0 = scaleFrame(f0,factor=0.5)
    cv2.imwrite("/tmp/todetect.jpg",f0)
    framenum = 0
    loop = asyncio.get_event_loop()
    detectFuture = loop.run_in_executor(None, detect, "/tmp/todetect.jpg")
    flow = FlowModel(f0)

    bgExtractor = BackgroundExtractor()
    def updateLines(que):
        while len(que)> 0:
            flow.apply(que.popleft())
        model = flow.getModel()
        parallel,front = getClassified(background,model)
        return parallel, front

    frameque = deque([],60)
    for i in range(100):
        print(i)
        r0,f0 = cap.read()
        f0 = scaleFrame(f0,factor=0.5)
        frameque.append(f0)
        cv2.imshow("fg",bgExtractor.apply(f0))
        cv2.imshow("bg", bgExtractor.getBackground())
        cv2.waitKey(20)
    bgFuture = loop.run_in_executor(None, bgExtractor.apply, f0)
    linesFuture = loop.run_in_executor(None, updateLines, frameque)
    # print(detectFuture.done())
    initiated = False
    while True:
        await asyncio.sleep(0)
        ret, frame = cap.read()
        framenum+=1
        if ret:
            frame = scaleFrame(frame,factor=0.5)
            # bgExtractor.apply(frame)
            # background = bgExtractor.getBackground()
            # fmodel = flow.apply(frame)
            # parallel,front = getClassified(background,fmodel)

            cv2.imwrite("/tmp/todetect.jpg",frame)

            if(linesFuture.done()):
                if(len(frameque) < 60):
                    frameque.append(frame)
                else:
                    lineFuture = loop.run_in_executor(None, updateLines, frameque)

            if(bgFuture.done()):

                bgFuture.cancel()
                cv2.imshow("bg", bgExtractor.getBackground())
                bgFuture = loop.run_in_executor(None, bgExtractor.apply, frame)

            if(detectFuture.done()):
                initiated = True
                print("result")
                objs = detectFuture.result()
                dets = detsYoloToSortInput(objs)
                trackers,hist = motTracker.update(dets,True )
                tracks = historyToTracks(hist)
                detectFuture.cancel()
                detectFuture = loop.run_in_executor(None, detect, "/tmp/todetect.jpg")

            if initiated:
                drawSortDetections(trackers, frame)
                # drawSortHistory(hist, frame)
                # pol = historyToPolylines(hist)
                for trk in tracks:
                    # print(trk)
                    cv2.polylines(frame, [trk[0]], False, colours[trk[1]%32,:])
                # cv2.polylines(frame, pol, False, (0,255,0))
                # draw_lines(frame,parallel, color=(0,0,255))
                # draw_lines(frame,front, color=(0,255,0))

            print(framenum)
            cv2.imshow('frame',frame)
        else:
            print("noit ok")
        k = cv2.waitKey(30) & 0xff

loop.run_until_complete(main())

print("complete")
cap.release()
cv2.destroyAllWindows()
