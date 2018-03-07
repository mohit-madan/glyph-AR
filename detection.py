import numpy as np
#import extractMatrix
import order_pts
import cv2
i=0

cam = cv2.VideoCapture(0)

def order_pts(pts):
    s = np.sum(pts,axis = 1)
    ord_pts = np.zeros((4,2), dtype = "float32")
    ord_pts[0] = pts[np.argmin(s)]
    ord_pts[2] = pts[np.argmax(s)]

    diff = np.diff(pts,axis = 1)
    ord_pts[1] = pts[np.argmin(diff)]
    ord_pts[3] = pts[np.argmax(diff)]
    return ord_pts

def extractMatrix(image,pts):
    pts = order_pts(pts)
    print(pts)
    (tl,tr,br,bl) = pts

    #compute max height 
    h1 = np.sqrt(np.square(tl[1] - bl[1]) + np.square(tl[0] - bl[0]))
    h2 = np.sqrt(np.square(tr[1] - br[1]) + np.square(tr[0] - br[0]))

    #width
    w1 = np.sqrt(np.square(tl[1] - tr[1]) + np.square(tl[0] - tr[0]))
    w2 = np.sqrt(np.square(bl[1] - tr[1]) + np.square(tl[0] - tr[0]))

    newHeight = max(h1,h2)
    newWidth = max(w1,w2)

    newPoints = np.array([[0,0],[newWidth-1,0],[newWidth-1,newHeight-1],[0,newHeight-1]],dtype = "float32")
    print(newPoints)
    print(pts)
    H = cv2.getPerspectiveTransform(pts,newPoints)
    warped_img = cv2.warpPerspective(image,H,(newWidth,newHeight))

    return warped_img,H

while(True):

    # Capture frame-by-frame
    ret, frame = cam.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 	
 	# create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray) #applying histogram equalisation
    
    gray = cv2.GaussianBlur(gray, (5,5), 0) #gaussian blur-to smoothen out random edges 
    gray_edge = cv2.Canny(gray,100,300) #applying canny edge detection

    im2, contours, _ = cv2.findContours(gray_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  #finding contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] #sorting contours in reverse order - why? don't know    
    #approxiamating contours 

    for cnt in contours:
        #We will find if each of the detected contour is of quad shape, then we will do the perspective
        #transform of the image to get the quad in top down view and then the glyph detction lago will prceed
        epsilon = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,0.01*epsilon,True) #with greater percentage a large set is coming
        #cv2.drawContours(frame, cnt, -1, (0,255,0), 3)
        cv2.drawContours(frame, approx, -1, (0, 0, 255), 3)
        cv2.imshow('frame',frame)
        vert_num = len(approx)
        if vert_num ==4:
            i=i+1   
            print(i)
            approx = approx.reshape(4,2)
            print(approx)
            #warped_img , H = extractMatrix(gray,approx)
            #cv2.imshow("original",gray)
            #cv2.imshow("transformed",warped_img)
            
    

    
    # if vert==4:
    #   break
    # # Display the resulting frame
        

    


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()


 
#for contour in contours:
