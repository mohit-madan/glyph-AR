import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 	
 	# create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray) #applying histogram equalisation
    
    gray = cv2.GaussianBlur(gray, (5,5), 0) #gaussian blur
    gray_edge = cv2.Canny(gray,100,300) #applying canny edge detection

    im2, contours, _ = cv2.findContours(gray_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  #finding contours
        
    #approxiamating contours 
    # cnt = contours[0]
    # epsilon = 0.1*cv2.arcLength(cnt,True)
    # approx = cv2.approxPolyDP(cnt,epsilon,True)
    # print(approx)
    
    # Display the resulting frame
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)    
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


 
#for contour in contours:
