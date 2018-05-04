# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# https://rdmilligan.wordpress.com/2015/06/28/opencv-camera-calibration-and-pose-estimation-using-python/

from webcam import Webcam
import cv2
from datetime import datetime

webcam = Webcam()
webcam.start()

while True:

    # get image from webcam
    image = webcam.get_current_frame()

    # display image
    cv2.imshow('grid', image)
    cv2.waitKey(3000)

    # save image to file, if pattern found
    ret, corners = cv2.findChessboardCorners(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), (7,6), None)

    if ret == True:
        print('success')
        filename = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg'
        cv2.imwrite("pose/sample_images/" + filename, image)