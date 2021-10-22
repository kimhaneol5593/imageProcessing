import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
def houghCircle(img): 
    # 원이 있는 이미지를 올림. 
    img1 = img 
    img2 = img1.copy() 
    
    # 0~9까지 가우시안 필터로 흐리게 만들어 조절함. 
    img2 = cv.GaussianBlur(img2, (9, 9), 0) 
    # 그레이 이미지로 바꿔서 실행해야함. 
    imgray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) 
    # 원본과 비율 / 찾은 원들간의 최소 중심거리 / param1, param2를 조절해 원을 찾음 
    circles = cv.HoughCircles(imgray, cv.HOUGH_GRADIENT, 1, 10, param1=60, param2=50, minRadius=0, maxRadius=0) 
    if circles is not None: 
        circles = np.uint16(np.around(circles)) 
        print(circles) 
        for i in circles[0, :]:
            cv.circle(img1, (i[0], i[1]), i[2], (255, 255, 0), 2)
            

        cv.imshow('HoughCircle', img1) 
        cv.waitKey(0) 
        cv.destroyAllWindows()
    else: 
        print('원을 찾을 수 없음') 
flag = ' '
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # # define range of blue color in HSV
    # lower_blue = np.array([110,50,50])
    # upper_blue = np.array([130,255,255])
    
    # 초록색
    lower_green = np.array([50, 150, 50])
    upper_green = np.array([80, 255, 255])

    #빨간색
    lower_red = np.array([150, 50, 50])
    upper_red = np.array([180, 255, 255])


    # Threshold the HSV image to get only blue colors
    # mask = cv.inRange(hsv, lower_blue, upper_blue)
    mask1 = cv.inRange(hsv, lower_green, upper_green)
    mask2 = cv.inRange(hsv, lower_red, upper_red)


    # Bitwise-AND mask and original image
    # res = cv.bitwise_and(frame,frame, mask= mask)
    res1 = cv.bitwise_and(frame, frame, mask=mask1)
    res2 = cv.bitwise_and(frame, frame, mask=mask2)

    houghCircle(res2)
    cv.imshow('frame',frame)
    cv.imshow('green', res1)
    cv.imshow('red', res2)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()



