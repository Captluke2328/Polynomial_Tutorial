import cv2

img_path = r"/Users/jlukas/Desktop/My_Project/Polynomial_Tutorial/Basket_Ball_Shot_Predictor/Ball.png"

while True:
    '''Grab the Image'''
    #success, img = cap.read()
    img = cv2.imread(img_path)

    '''Crop the image height - 0 -> 1080 but we crop it to 0 -> 900'''
    img = img[0:900,:]

    '''Display'''
    img = cv2.resize(img, (0,0), None, 0.7,0.7)
    cv2.imshow("Result",img)
    if cv2.waitKey(100) and 0XFF == ord('q'):
        break
