import cv2
import cvzone
from cvzone.ColorModule import ColorFinder

img_path = r"/Users/jlukas/Desktop/My_Project/Polynomial_Tutorial/Basket_Ball_Shot_Predictor/Ball.png"
vid_path = r"/Users/jlukas/Desktop/My_Project/Polynomial_Tutorial/Basket_Ball_Shot_Predictor/Videos/vid (3).mp4"

'''Intialize the Video'''
cap = cv2.VideoCapture(vid_path)

'''Create the color Finder Object True: Debug Mode, False : Non Debug Mode'''
myColorFinder = ColorFinder(False)
hsvVals = {'hmin' : 0, 'smin' : 152, 'vmin' : 0, 'hmax' : 91, 'smax' : 255, 'vmax' : 255}

'''Variables'''
posList = []

while True:
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
        posList.append(contours[0]['center'])
        #cx , cy = contours[0]['center']
        
    for i, (cx,cy) in enumerate(posList):
        cv2.circle(imgContours, (cx,cy), 5, (0,255,0), cv2.FILLED)
        
        if i == 0:
            cv2.line(imgContours, (cx,cy), (cx,cy), (0,255,0),3)
           
        else: 
            cv2.line(imgContours, (cx,cy), tuple(posList[i-1]), (0,255,0),3)
        
        # poslist consist of the previous coordinate of cx,cy ball.
        print(cx,cy, posList[i-1])
    
    '''Display'''
    imgContours = cv2.resize(imgContours, (0,0), None, 0.7,0.7)
    #cv2.imshow("Result",img)
    cv2.imshow("Result",imgContours)
    #cv2.imshow("Result",mask)

    if cv2.waitKey(50) and 0XFF == ord('q'):
        break
