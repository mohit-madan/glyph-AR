import numpy as np
import cv2
i=0

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
    for cnt in contours:
        #We will find if each of the detected contour is of quad shape, then we will do the perspective
        #transform of the image to get the quad in top down view and then the glyph detction lago will prceed
        epsilon = 0.01*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,False)
        cv2.drawContours(frame, cnt, -1, (0,255,0), 3)
        cv2.drawContours(frame, approx, -1, (0, 0, 255), 3)
        vert = len(approx)
        if vert ==4:
            print(vert)
            i=i+1   
            print(i)
            approx = approx.reshape(4,2)
            print(approx)
            break

 #   if vert ==4:
  #      break

    
    # Display the resulting frame
        

    cv2.imshow('frame',frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


 
#for contour in contours:
