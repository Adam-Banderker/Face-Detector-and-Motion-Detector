import cv2




while True:
  
    # Get user supplied values
    # User interaction for input function
    inputimage = input("Please insert image name: ").lower()
    imagePath = str(inputimage)
    cascPath = "haarcascade_frontalface_default.xml"
    casceyePath = "haarcascade_eye.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)
    eye_cascade = cv2.CascadeClassifier(casceyePath)


    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    #
    if len(faces) > 1:
        print("Found {0} faces!".format(len(faces)))
    else:
        print("Found {0} face!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor = 1.3)
        
        if len(eyes)<2:
            print("Found {0} eye! You are an alien".format(len(eyes)))
        else:
            print("Found {0} eyes! You are a Human".format(len(eyes)))

        # Draw a rectangle around the eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    #resized =cv2.resize(image,(int(image.shape[1]/7),int(image.shape[0]/7)))
    #resized =cv2.resize(image,(int(image.shape[1]/2),int(image.shape[0]/2)))
    resized =cv2.resize(image,(int(image.shape[1]),int(image.shape[0])))

    
    cv2.imshow("Faces found", resized)
    k = cv2.waitKey(0)
    if k == 27:
        break
    else:
        continue

cv2.destroyAllWindows()

    

