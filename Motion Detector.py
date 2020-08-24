import cv2
import numpy as np

# Video Capture

capture = cv2.VideoCapture(0)

#History, Threshold, DetectShadows

fgbg = cv2.createBackgroundSubtractorMOG2(50, 200, True)

#Keep track of what frame we're on
frameCount = 0

while True:
    #Return Value and the current Frame
    ret,frame = capture.read()

    # Check if a current frame actually exists
    if not ret:
        break

    frameCount += 1

    # Resize the frame
    resizedFrame = cv2.resize(frame, (0,0), fx = 0.80, fy = 0.80)

    # Get the foreground Mask
    fgmask = fgbg.apply(resizedFrame)

    # Count all the none zero pixels within the mask
    count = np.count_nonzero(fgmask)

    print(" Frame: %d, Pixel Cout: %d" % (frameCount, count))

    if (frameCount > 1 and count > 1000):
        print("Alert! Someone is there!")
        cv2.putText(resizedFrame, "Alert! Someone is there!", (10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)

    cv2.imshow("Frame", resizedFrame)
    cv2.imshow("Mask", fgmask)

    k = cv2.waitKey(1)
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()
    
