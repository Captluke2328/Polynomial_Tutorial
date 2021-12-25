from tracemalloc import start
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

img_path = r"/Users/jlukas/Desktop/My_Project/Polynomial_Tutorial/Basket_Ball_Shot_Predictor/Ball.png"
vid_path = r"/Users/jlukas/Desktop/My_Project/Polynomial_Tutorial/Basket_Ball_Shot_Predictor/Videos/vid (2).mp4"

'''Intialize the Video'''
cap = cv2.VideoCapture(vid_path)

'''Create the color Finder Object True: Debug Mode, False : Non Debug Mode'''
myColorFinder = ColorFinder(False)
hsvVals = {'hmin' : 0, 'smin' : 152, 'vmin' : 0, 'hmax' : 91, 'smax' : 255, 'vmax' : 255}

'''Variables'''
posListX = []
posListY = []

'''Referring to image Width X that will be used to find Y value based on Polynomial Equation'''
xList = [item for item in range(0,1300)]

while True:
    if start: 
        '''Grab the Image'''
        success, img = cap.read()
        #img = cv2.imread(img_path)

        '''Crop the image height - 0 -> 1080 but we crop it to 0 -> 900'''
        img = img[0:900,:]

        '''Find the Color Ball'''
        imgColor, mask = myColorFinder.update(img,hsvVals)
        
        '''Find the image contour - Min area to be detected as Ball Pixel'''
        imgContours, contours = cvzone.findContours(img, mask, minArea=500)
        
        '''Print the point of ball direction movement'''
        if contours:
            '''Take the biggest contours as it is already sorted'''
            posListX.append(contours[0]['center'][0])
            posListY.append(contours[0]['center'][1])
            
            #cx , cy = contours[0]['center']
            
        if posListX:
            '''Polynomial Regression y = Ax^2 + Bx + C'''
            '''Find the Coefficients'''
            
            '''Below is Coefficients - Quadratic - Second Order '''
            A,B,C = np.polyfit(posListX, posListY,2)

            '''Here we zip the posListX and Y because we will running the for loop for both'''
            for i, (cx,cy) in enumerate(zip(posListX, posListY)):
                cv2.circle(imgContours, (cx,cy), 8, (0,255,0), cv2.FILLED)
                
                if i == 0:
                    cv2.line(imgContours, (cx,cy), (cx,cy), (0,255,0),3)
                
                else: 
                    #cv2.line(imgContours, (cx,cy), tuple(posList[i-1]), (0,255,0),3)
                    cv2.line(imgContours, (cx,cy), (posListX[i-1], posListY[i-1]), (0,255,0),3)
                
                # poslist consist of the previous coordinate of cx,cy ball.
                # print(cx,cy, posList[i-1])
            
            for x in xList:
                y = int(A * x ** 2 + B * x + C)
                cv2.circle(imgContours, (x,y), 2, (255,0,255), cv2.FILLED)
            cv2.putText(imgContours, ("Projection :" + str(y)), (10,50),cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 4)
 
        '''Display'''
        imgContours = cv2.resize(imgContours, (0,0), None, 0.7,0.7)
        #cv2.imshow("Result",img)
        cv2.imshow("Polynomial Regression - Ball Shot Predictor",imgContours)
        #cv2.imshow("Result",mask)

    key = cv2.waitKey(100)
    if key == ord("s"):
        start = False
    if key == ord("q"):
        start = True

