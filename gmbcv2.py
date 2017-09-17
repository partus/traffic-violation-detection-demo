import numpy as np
import cv2
cap = cv2.VideoCapture("/data/livetraffic/2017-08-27/3/tokyo.mp4")


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=200,useHistory=True,isParallel=True)
# Docstring:
# createBackgroundSubtractorCNT([, minPixelStability[, useHistory[, maxPixelStability[, isParallel]]]]) -> retval
# .   @brief Creates a CNT Background Subtractor
# .
# .   @param minPixelStability number of frames with same pixel color to consider stable
# .   @param useHistory determines if we're giving a pixel credit for being stable for a long time
# .   @param maxPixelStability maximum allowed credit for a pixel in history
# .   @param isParallel determines if we're parallelizing the algorithm
# Type:      builtin_function_or_method
#
framenum = 0
while(1):
    framenum+=1
    ret, frame = cap.read()
    if not framenum % 100:
        print(framenum)
    if ret:
        height,width, layers = frame.shape
        frame = cv2.resize(frame, (int(width/2),int(height/2)))
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        cv2.imshow('fgmask',fgmask)
        cv2.imshow('frame',frame)
    # else:
    #     cv2.waitKey(1000)
    #     break
    if 0xFF & cv2.waitKey(10) == 27:
        break
cap.release()
cv2.destroyAllWindows()
