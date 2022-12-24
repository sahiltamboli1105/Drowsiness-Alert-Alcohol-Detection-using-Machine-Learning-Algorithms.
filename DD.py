import tkinter as tk
from tkinter import *
from PIL import Image , ImageTk


root = tk.Tk()
w , h = root.winfo_screenwidth() , root.winfo_screenheight()
root.geometry("%dx%d+0+0"%(w,h))
img = Image.open("bg.jpg")
img =img.resize((w,h), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(img)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0) #, relwidth=1, relheight=1)




def Drowsiness():
    import cv2
    import numpy as np
    import time
    import sys

    path = "classifiers/haar-face.xml"
    faceCascade = cv2.CascadeClassifier(path)

    # Variable used to hold the ratio of the contour area to the ROI
    ratio = 0

    # variable used to hold the average time duration of the yawn
    global yawnStartTime
    yawnStartTime = 0

    # Flag for testing the start time of the yawn
    global isFirstTime
    isFirstTime = True

    # List to hold yawn ratio count and timestamp
    global yawnRatioCount
    yawnRatioCount = []

    # Yawn Counter
    yawnCounter = 0

    # yawn time
    averageYawnTime = 2.5

    """
    Find the second largest contour in the ROI; 
    Largest is the contour of the bottom half of the face.
    Second largest is the lips and mouth when yawning.
    """

    def calculateContours(image, contours):
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        maxArea = 0
        secondMax = 0
        maxCount = 0
        secondmaxCount = 0
        for i in contours:
            count = i
            area = cv2.contourArea(count)
            if maxArea < area:
                secondMax = maxArea
                maxArea = area
                secondmaxCount = maxCount
                maxCount = count
            elif (secondMax < area):
                secondMax = area
                secondmaxCount = count

        return [secondmaxCount, secondMax]

    """
    Thresholds the image and converts it to binary
    """

    def thresholdContours(mouthRegion, rectArea):
        global ratio

        # Histogram equalize the image after converting the image from one color space to another
        # Here, converted to greyscale
        imgray = cv2.equalizeHist(cv2.cvtColor(mouthRegion, cv2.COLOR_BGR2GRAY))

        # Thresholding the image => outputs a binary image.
        # Convert each pixel to 255 if that pixel each exceeds 64. Else convert it to 0.
        ret, thresh = cv2.threshold(imgray, 64, 255, cv2.THRESH_BINARY)

        # Finds contours in a binary image
        # Constructs a tree like structure to hold the contours
        # Contouring is done by having the contoured region made by of small rectangles and storing only the end points
        # of the rectangle
        cnt, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        returnValue = calculateContours(mouthRegion, contours)

        # returnValue[0] => secondMaxCount
        # returnValue[1] => Area of the contoured region.
        secondMaxCount = returnValue[0]
        contourArea = returnValue[1]

        ratio = contourArea / rectArea

        # Draw contours in the image passed. The contours are stored as vectors in the array.
        # -1 indicates the thickness of the contours. Change if needed.
        if (isinstance(secondMaxCount, np.ndarray) and len(secondMaxCount) > 0):
            cv2.drawContours(mouthRegion, [secondMaxCount], 0, (255, 0, 0), -1)

    """
    Isolates the region of interest and detects if a yawn has occured. 
    """

    def yawnDetector(video_capture):
        global ratio, yawnStartTime, isFirstTime, yawnRatioCount, yawnCounter
        yawnCounter = 0
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            # flags=cv2.cv2.CV_HAAR_SCALE_IMAGE
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Isolate the ROI as the mouth region
            widthOneCorner = int((x + (w / 4)))
            widthOtherCorner = x + int(((3 * w) / 4))
            heightOneCorner = y + int((11 * h / 16))
            heightOtherCorner = y + h

            # Indicate the region of interest as the mouth by highlighting it in the window.
            cv2.rectangle(frame, (widthOneCorner, heightOneCorner), (widthOtherCorner, heightOtherCorner), (0, 255, 0),
                          2)
            # cv2.circle(img, center, radius, (0, 255, 0), 2)

            # mouth region
            mouthRegion = frame[heightOneCorner:heightOtherCorner, widthOneCorner:widthOtherCorner]

            # Area of the bottom half of the face rectangle
            rectArea = (w * h) / 2

            if (len(mouthRegion) > 0):
                thresholdContours(mouthRegion, rectArea)

            print("Current probablity of yawn: " + str(round(ratio * 1000, 2)) + "%")
            print("Length of yawnCounter: " + str(len(yawnRatioCount)))

            if (ratio > 0.06):
                if (isFirstTime is True):
                    isFirstTime = False
                    yawnStartTime = time.time()

                # If the mouth is open for more than 2.5 seconds, classify it as a yawn
                if ((time.time() - yawnStartTime) >= averageYawnTime):
                    yawnCounter += 1
                    yawnRatioCount.append(yawnCounter)

                    if (len(yawnRatioCount) > 8):
                        # Reset all variables
                        isFirstTime = True
                        yawnStartTime = 0
                        return True

        # Display the resulting frame
        cv2.namedWindow('yawnVideo')
        cv2.imshow('yawnVideo', frame)
        time.sleep(0.025)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(0)

        return False

    """
    Main
    """

    # Capture from web camera
    # yawnCamera = cv2.VideoCapture(0)

    import cv2
    import os
    from keras.models import load_model
    import numpy as np
    from pygame import mixer
    import time


    mixer.init()
    sound = mixer.Sound('alarm.wav')

    face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
    leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')



    lbl=['Close','Open']

    model = load_model('models/cnnCat2.h5')
    path = os.getcwd()
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    count=0
    score=0
    thicc=2
    rpred=[99]
    lpred=[99]
    #sound = mixer.Sound('alarm.wav')
    while(True):
        returnValue = (yawnDetector(cap), 'yawn')
        if returnValue[0]:
            print(returnValue)
            print("Yawn detected!")

            sound.play()
            # When everything is done, release the capture
            # yawnCamera.release()
            #cv2.destroyWindow('yawnVideo')
            #return returnValue
        else:
            sound.stop()




        ret, frame = cap.read()
        height,width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

        for (x,y,w,h) in right_eye:
            r_eye=frame[y:y+h,x:x+w]
            count=count+1
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye=  r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = model.predict_classes(r_eye)
            if(rpred[0]==1):
                lbl='Open'
            if(rpred[0]==0):
                lbl='Closed'
            break

        for (x,y,w,h) in left_eye:
            l_eye=frame[y:y+h,x:x+w]
            count=count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = model.predict_classes(l_eye)
            if(lpred[0]==1):
                lbl='Open'
            if(lpred[0]==0):
                lbl='Closed'
            break

        if(rpred[0]==0 and lpred[0]==0):
            score=score+1
            cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            print("Wake Up!!!")
        # if(rpred[0]==1 or lpred[0]==1):
        else:
            score=0
            cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            print("Safe!")

        if(score<0):
            score=0
        cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        if(score>5):
            #person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(path,'image.jpg'),frame)
            try:
                sound.play()

            except:  # isplaying = False
                pass
            if(thicc<16):
                thicc= thicc+2
            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
        else:
            sound.stop()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    #while True:
        #returnValue = (yawnDetector(cap), 'yawn')
        #if returnValue[0]:
            #print("Yawn detected!")
            # When everything is done, release the capture
            #yawnCamera.release()
           # cv2.destroyWindow('yawnVideo')
            #return returnValue

    #main()

    cap.release()
    cv2.destroyAllWindows()











#Yawn()
#EDrowsiness()

def Exit():
    root.destroy()
    


button1 = tk.Button(root,text="Drowsiness", command= Drowsiness)
button1.place(x=100 , y=100)

button2 =tk.Button(root,text="Exit", command= Exit)
button2.place(x=100 , y=200)

root.mainloop()