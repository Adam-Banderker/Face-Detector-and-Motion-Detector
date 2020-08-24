import cv2


# Get user supplied values
cascPath = "haarcascade_frontalface_default.xml"
casceyePath = "haarcascade_eye.xml"

# Create the haar cascade
face_cascade = cv2.CascadeClassifier(cascPath)
eye_cascade = cv2.CascadeClassifier(casceyePath)


cap = cv2.VideoCapture(0)



while True:
    # Read the image
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1,4)

    # Draw a rectangle around the faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0,0),0)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor = 1.3)

        # Draw a rectangle around the eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        


    cv2.imshow("img", img)

    k = cv2.waitKey(30)

    if k == 27:
        break

cap.release()
