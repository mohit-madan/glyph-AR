# Code according to OpenCV-Python Tutorial: Camera Calibration and 3D Reconstruction
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

import numpy as np
import glob
import cv2 as cv

# create grid for calibrating the camera
obj_pts = np.zeros((6 * 7, 3), np.float32)
obj_pts[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# storing the 3D points
object_points_array = []
# assuming all the points were visible initially
# store corresponding 2D image points for the same
image_points_array = []

# load images of chessboard
images = glob.glob('calibration_images/*.jpg')
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for fname in images:
    image = cv.imread(fname)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    # if successful in detecting the chessboard, draw the detected
    # corners using the built in function and store the image points
    if ret:
        object_points_array.append(obj_pts)

        cornersSubPixeled = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        image_points_array.append(cornersSubPixeled)

        image = cv.drawChessboardCorners(image, (7, 6), cornersSubPixeled, ret)
        cv.imshow('image', image)
        cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(object_points_array, image_points_array, gray.shape[::-1], None, None)
np.savez('camcalib.npz', ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
