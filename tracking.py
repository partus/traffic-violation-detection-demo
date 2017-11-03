from darknet import Detector

# print(detect("data/dog.jpg"))
import numpy as np
import cv2
import matplotlib.pylab as plt
from sort import Sort
import asyncio
from linetools import seg_poliline_intersect, draw_lines
from lsd import getClassified,getMainLines
from  backgroundExtraction import BackgroundExtractor
from functions import scaleFrame
from denseOpticalFlow import FlowModel
from collections import deque
import time
from videoWriter import VideoWriter

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
        cv2.rectangle(frame,(d[0],d[1]),(d[2],d[3]), colours[d[4]%32,:], thickness=2, lineType=8, shift=0)


def historyToPolylines(hist):
    ret = []
    for h in hist:
        trck = h[0]
        cont = []
        for rect in trck:
            cont.append((rect[:,0:2]+rect[:,2:4])/2)
        ret.append(np.array(cont, dtype=np.int32))
    return ret
def historyToTrack(h):
    trck = h[0]
    cont = []
    for rect,ts in trck:
        cont.append((rect[:,0:2]+rect[:,2:4])/2)
    return (np.array(cont, dtype=np.int32),h[1])


def historyToTracks(hist):
    ret = []
    for h in hist:
        ret.append(historyToTrack(h))
    return ret

def drawSortHistory(frame,history,violating=False,thickness=2,trackthickness=1):
    h = history
    color = colours[h[1]%32,:]
    if violating:
        color = (0,0,255)
    else:
        color = (0,255,0)
    trck = h[0]
    if(len(trck) > 0):
        rect = trck[-1][0]
        d = rect[0].astype(np.int32)
        cv2.rectangle(frame,(d[0],d[1]),(d[2],d[3]), color, thickness=thickness, lineType=8, shift=0)
        if(violating):
            cv2.line(frame, (d[0],d[1]),(d[2],d[3]),color, thickness=thickness, lineType=8, shift=0)
            cv2.line(frame, (d[0],d[3]),(d[2],d[1]),color, thickness=thickness, lineType=8, shift=0)

        trk = historyToTrack(history)
        cv2.polylines(frame, [trk[0]], False, color,thickness=trackthickness)

def drawSortHistory1(history, frame):
    for h in history:
        trck = h[0]
        color = colours[h[1]%32,:]
        for rect in trck:
            d = rect[0].astype(np.int32)
            cv2.rectangle(frame,(d[0],d[1]),(d[2],d[3]), color, thickness=3, lineType=8, shift=0)


# cap = cv2.VideoCapture("/data/livetraffic/2017-07-18/City of Auburn Toomer's Corner Webcam 2-yJAk_FozAmI.mp4")
# cap = cv2.VideoCapture("/data/Simran Official Trailer _ Kangana Ranaut _  Hansal Mehta _ T-Series-_LUe4r6eeQA.mkv")
# cap = cv2.VideoCapture("/data/livetraffic/2017-07-18/Jackson Hole Wyoming Town Square - SeeJH.com-psfFJR3vZ78.mp4")
# cap.set(cv2.CAP_PROP_POS_FRAMES, 80000)

motTracker = Sort(max_age=30,min_hits=4)
async def detectAsync():

    objs = detect("/tmp/todetect.jpg")
    return objs
    # return objs


# async def futurise():

class Executor:
    def __init__(self,func,*args):
        self.func = func
        loop = asyncio.get_event_loop()
        self.future = loop.run_in_executor(None, detect, *args)
    def res(self):
        if self.future.done():
            True

def _updateLines(que, flow, background):
    while len(que)> 0:
        flow.apply(que.popleft())
    model = flow.getModel()
    # background = bgExtractor.getBackground()
    allLines,parallel,front = getClassified(background,model)
    return allLines,parallel, front

def updateLines(que, flow, background):
    while len(que)> 0:
        que.popleft()
    # model = flow.getModel()
    # background = bgExtractor.getBackground()
    allLines = getMainLines(background)
    return allLines,[],[]
colorMap = {
    # BGR color space
    'P': (255,0,0),
    'R': (0,0,255),
    'Y': (0,255,255),
    'G': (0,255,0)
}
async def main(display,lineStorage,loop,scaleFactor=1,video="/data/livetraffic/2017-08-27/3/tokyo.mp4"):
    # cap = cv2.VideoCapture("/data/livetraffic/2017-07-18/City of Auburn Toomer's Corner Webcam 2-yJAk_FozAmI.mp4")
    cap = cv2.VideoCapture(video)
    cap = cv2.VideoCapture("/data/livetraffic/2017-07-18/taiwan.mp4")
    # cap = cv2.VideoCapture('/data/livetraffic/2017-07-18/La Grange, KY - Virtual Railfan LIVE (La Grange, KY North)-Bv3l77cRRGY.mp4')
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 80000)
    # cap = cv2.VideoCapture("/data/livetraffic/2017-07-18/Jackson Hole Wyoming Town Square - SeeJH.com-psfFJR3vZ78.mp4")
    r0,f0 = cap.read()
    f0 = scaleFrame(f0,factor=scaleFactor)
    cv2.imwrite("/tmp/todetect.jpg",f0)
    framenum = 0
    detectFuture = loop.run_in_executor(None, detect, "/tmp/todetect.jpg")
    flow = FlowModel(f0)

    bgExtractor = BackgroundExtractor()


    frameque = deque([],60)
    for i in range(4000):
        if not i % 40:
            print(i)
            r0,f0 = cap.read()
            f0 = scaleFrame(f0,factor=scaleFactor)
            if(r0):
                frameque.append(f0)
                bgExtractor.apply(f0)
        # cv2.imshow("fg",bgExtractor.apply(f0))
        # cv2.imshow("bg", bgExtractor.getBackground())
    bgFuture = loop.run_in_executor(None, bgExtractor.apply, f0)
    linesFuture = loop.run_in_executor(None, updateLines, frameque,flow,bgExtractor.getBackground())
    allLines,parallel,front =[],[],[]
    # print(detectFuture.done())
    initiated = False
    violationSaver = VideoWriter()
    while framenum < 30000:
        await asyncio.sleep(0.05)
        ret, frame = cap.read()
        framenum+=1
        if ret:
            frame = scaleFrame(frame,factor=scaleFactor)
            violationSaver.addFrame(frame)
            if(linesFuture.done()):
                if(len(frameque) < 60):
                    frameque.append(frame)
                else:
                    print("got lines", allLines)
                    allLines,parallel,front = linesFuture.result()
                    lineStorage.apply(allLines)
                    linesFuture.cancel()
                    linesFuture = loop.run_in_executor(None, updateLines, frameque,flow,bgExtractor.getBackground())
            if not framenum % 20:
                if(bgFuture.done()):
                    bgFuture.cancel()
                    # cv2.imshow("bg", bgExtractor.getBackground())
                    bgFuture = loop.run_in_executor(None, bgExtractor.apply, frame)

            if(detectFuture.done()):
                initiated = True
                print("result")
                objs = detectFuture.result()
                dets = detsYoloToSortInput(objs)
                trackers,hist = motTracker.update(dets,getHistory=True )
                tracks = historyToTracks(hist)
                detectFuture.cancel()
                cv2.imwrite("/tmp/todetect.jpg",frame)
                detectFuture = loop.run_in_executor(None, detect, "/tmp/todetect.jpg")

            if initiated:
                # drawSortDetections(trackers, frame)
                # drawSortHistory(hist, frame)
                groups = lineStorage.getGroups()
                for h in hist:
                    violating = False
                    trk = historyToTrack(h)

                    for id,group in groups.items():
                        if group['type'] in ['P','R']:
                            intersects = seg_poliline_intersect(group['main'],trk[0])
                            if(len(intersects) > 0):
                                violating = True
                    if(violating):
                        violationSaver.saveViolation(h[1])

                    drawSortHistory(frame,h,violating=violating)
                # for trk in tracks:
                #     # print(trk)
                #     cv2.polylines(frame, [trk[0]], False, colours[trk[1]%32,:],thickness=2)
                #     # for line in allLines:
                #     #     intersects = seg_poliline_intersect(line,trk[0])
                #     #     for intersect in intersects:
                #     #         cv2.circle(frame, tuple(intersect.astype(np.uint32)), 5, (0,0,255), thickness=3, lineType=8, shift=0)
                #
                #     for id,group in groups.items():
                #         intersects = seg_poliline_intersect(group['main'],trk[0])
                #         for intersect in intersects:
                #             cv2.circle(frame, tuple(intersect.astype(np.uint32)), 5, (0,0,255), thickness=3, lineType=8, shift=0)

                # cv2.polylines(frame, pol, False, (0,255,0))
                print("parallel")
                print(parallel)
                draw_lines(frame,allLines, color=(255,255,0))
                # draw_lines(frame,parallel, color=(255,0,255))
                # draw_lines(frame,front, color=(0,255,0))

                for id,group in groups.items():
                    print(group,group['type'])
                    draw_lines(frame,[group['main']], color=colorMap[group['type']],thickness=3)
                    draw_lines(frame,group['other'], color=colorMap[group['type']],thickness=3)
            print(framenum)
            display(frame)
        else:
            print("noit ok")
        cv2.waitKey(1)

class Tracking:
    def __init__(self,loop,display,lineStorage,video,factor):
        self.display = display
        self.lineStorage = lineStorage
        self.loop = loop
        self.video = video
        self.factor = factor
    async def __call__(self):
        await main(self.display,self.lineStorage,self.loop, scaleFactor=self.factor,video=self.video)
    def stop(self):
        self.loop.close()

def tracking():
    loop = asyncio.get_event_loop()
    def display(frame):
        cv.imshow('frame',frame)
    loop.run_until_complete(main(display))
    print("complete")
    cap.release()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    def display(frame):
        cv2.imshow('frame',frame)
    loop.run_until_complete(main(display))
    print("complete")
    cap.release()
# cv2.destroyAllWindows()
