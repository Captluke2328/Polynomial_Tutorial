import cv2
frameWidth = 640
frameHeight = 480

path = r"/Users/jlukas/Desktop/My_Project/Polynomial_Tutorial/Basket_Ball_Shot_Predictor/Videos/vid (1).mp4"

cap = cv2.VideoCapture(path)

while (cap.isOpened()):
    success, img = cap.read()
    '''Here we define exact value'''
    #img = cv2.resize(img, (frameWidth, frameHeight))

    '''Here we set the scale insted of using exact value'''
    img = cv2.resize(img, (0,0), None, 0.7,0.7)
    cv2.imshow("Result", img)

    # 1 is too Fast we have to set to something larger like 100 for slower movement
    if cv2.waitKey(100) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()