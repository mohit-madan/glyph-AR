# Code according to Electric Soup Blog: OpenCV Camera Calibration and Pose Estimation using Python
# https://rdmilligan.wordpress.com/2015/06/28/opencv-camera-calibration-and-pose-estimation-using-python/

# Instructions: print out chessboard pattern on paper or display e.g. on phone
# Make it visible in frame
# Change position when "success" is printed

import cv2
from datetime import datetime

cam = cv2.VideoCapture(0)

while True:

    # get image from webcam
    ret, image = cam.read()

    # display image
    cv2.imshow('webcam', image)
    # wait enough time to change pose of chessboard before taking next image
    cv2.waitKey(3000)

    # save image to file, if pattern found
    ret, corners = cv2.findChessboardCorners(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (7, 6), None)

    if ret:
        print('success')
        filename = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg'
        cv2.imwrite("calibration_images/" + filename, image)
