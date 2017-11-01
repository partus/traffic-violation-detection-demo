# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

import numpy as np
import cv2

from collections import deque

def junk():
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame,0)

            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

class VideoWriter:
    def __init__(self):
        self.fourcc = fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.toSave = {}
        frameSec = 24
        self.frameCount = frameSec*20
        self.frameForward = frameSec*10
        self.frameNum = 0
        self.storage = deque([],self.frameCount)
    def saveViolation(self,trckNum):
        if trckNum in self.saving:
            return True
        else:
            self.saving[trckNum] = self.frameNum
    def saveVideo(self,frameList,name):
        filename = str(name)+".avi"
        # out = cv2.VideoWriter(filename,self.fourcc, 20.0, (640,480))
    def addFrame(self,frame):
        self.frameNum+=1
        self.storage.append(frame)
        for key,value in self.toSave.items():
            if(value + self.frameForward >= self.frameNum):
                frameList = list(self.storage)
                del self.toSave[key]
                self.saveVideo(frameList, value)
