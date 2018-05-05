

import numpy as np
import glob 
import cv2 as cv 

#let us create some points on the object for calbrating the camera 
obj_pts = np.zeros((6*7,3), np.float32)
obj_pts[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

#stroing the 3D points 
object_points_array = []
#assuming all the points were visible intially 
#store corresponding 2d image points for the same 
image_points_array = []

#load all teh images with fixed content ,
images = glob.glob('pose/sample_images/*.jpg')
criteria = (cv2.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for fname in images:
    image = cv.imread(fname)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    ret,corners = cv.findChessboardCorners(gray,(7,6),None)
    #if u are succesful in detecting the chess boarrd, draw the detetcted 
    #corners using the built in function and store the image points 
    if ret=True:
        object_points_array.append(obj_pts)

        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        image_points_array.append(corners2)     

        image = cv.drawChessboardCorners(image, (7, 6), corners2, ret)
        cv.imshow('image', image)
        cv.waitKey(500)

 cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(object_points, impoints, gray.shape[::-1], None, None)
np.savez('camcalib.npz', ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)