import numpy as np
import cv2

cap = cv2.VideoCapture("/data/livetraffic/2017-08-27/3/tokyo.mp4")

fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    if ret:
        fgmask = fgbg.apply(frame)

        cv2.imshow('frame',fgmask)
        cv2.imshow('mog',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
