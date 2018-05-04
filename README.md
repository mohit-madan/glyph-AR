# glyph-AR
We have implemented glyph detection for augmented reality applications. We detect glyphs(a playcard, a hieroglyphic symbol) and replace them with a 3D object using VTK and OpenCV. 

# Motivation 
1) Glyphs can be detected in videos and then substitued with 3D Virtual objects creating a view which is half real and half virtual and hence interfacing the virtual world with the real.
2) Glyphs also find application in robotics where robots can be navigated using glyphs. 

# Solution Approach
Our algorithm can be broken down into 3 main parts:-
1) Preprocessing : Apply blur, histogram equalization and detect contours to get the glyph in top-down view. Use homography for the same (top-down view necessary to decode the glyph)
2) Project a 2D image on the detected glyph. To stabilize the projection either use a KLT tracker or use a counter to increase the duration for which image is projected for each match 
3) Most tricky part is to project a 3D object. We imitate the the webcam by extracting its intrinsics using a chess board. We use this camera calibration matrix for finding out the correspondacen between 3D object 2Dimage plane.

# Dependencies Required to Run the Code 
$ sudo apt-get install python-vtk
$ sudo apt-get install python-pygame
$ sudo apt-get install python-opencv
$ sudo pip install openpyxl
$ pip install opencv-contrib-python
$ sudo apt-get install python-pip  
$ sudo pip install numpy scipy

# Running the code
To run the AR appllication , just type in the terminal "glyph_detection_main.py". Bring various glyphs in front of the camera and try to see the results. The glyph detection part is very sensitive to lighting conditions and hence you may want to ensure proper lights on the glyph.

# Results 
We were able to succesfully implement 3D object overaly over glyphs. One needs to be patient while working with VTK ( Python wrapper for OpenGL). 
