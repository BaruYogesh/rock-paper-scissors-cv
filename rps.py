import cv2
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)

pointA = (0,0)
pointB = (200,200)

#look for values between these two.
lower_skin = np.array([0,20,70], dtype=np.uint8)
upper_skin = np.array([20,255,255], dtype=np.uint8)


def caption(frame, text, )

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if (ret):

        #flip and get kernel        
        frame = cv2.flip(frame, 1)
        kernel = np.ones((3,3), np.uint8)

        #define region of interest and draw rectangle
        roi = frame[pointA[0]:pointB[0], pointA[1]:pointB[1]]
        cv2.rectangle(frame, pointA, pointB, (0,255,0), 0)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        #create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.dilate(mask, kernel, iterations = 4)
        mask = cv2.GaussianBlur(mask, (5,5), 100)

        #find contours
        contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        cnt = max(contours,key = lambda x: cv2.contourArea(x))

        epsilon = .0005*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        hull = cv2.convexHull(cnt)

        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)

        arearatio = ((areahull-areacnt)/areacnt) * 100

        hull = cv2.convexHull(approx, returnPoints = False)
        defects = cv2.convexityDefects(approx, hull)

        
        #draw lines around hand
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])

            cv2.line(roi,start, end, [0,255,0], 2)

        if areacnt<2000:
            cv2.putText(frame, 'Put hand in the box', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    

        cv2.imshow('frame',frame)
        cv2.imshow('mask', mask)

        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cv2.destroyAllWindows()
cap.release()