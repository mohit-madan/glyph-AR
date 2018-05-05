# Glyph-AR
We have implemented glyph detection for augmented reality applications. We detect glyphs(a playcard, a hieroglyphic symbol) and replace them with a 3D object using VTK and OpenCV. 

# Motivation 
1) Glyphs can be detected in videos and then substituted with 3D Virtual objects creating a view which is half real and half virtual and hence interfacing the virtual world with the real.
2) Glyphs also find application in robotics where robots can be navigated using glyphs. 

# Solution Approach
Our algorithm can be broken down into 3 main parts:-
1) Preprocessing : Apply blur, histogram equalization and detect contours in which glyph may be present. Filter out maximum unwanted quadrilaterals which are not glyphs before feeding them to glyph_detection algorithm to make the code robust and fast.
2) Use homography to get the top-down view of the glyph and then decode the glyph to project an appropriate 2D image on it.To stabilize the projection either use a KLT tracker or use a counter to increase the duration for which image is projected for each match.
3) Most tricky part is to project a 3D object. We imitate the the webcam by extracting its intrinsics by calibrating it beforehand (using a chess board). We use this extracted camera calibration matrix for finding out the correspondacen between 3D object to 2D image plane.

# Dependencies Required to Run the Code
Python 2.7
```
$ sudo apt-get install python-vtk
$ sudo apt-get install python-pygame
$ sudo apt-get install python-opencv
$ sudo pip install openpyxl
$ pip install opencv-contrib-python
$ sudo apt-get install python-pip  
$ sudo pip install numpy scipy
```
see requirements.txt

# Running the code
#### 3D AR - Superpose different cubes on glyphs
To run the AR appllication , just type in the terminal "glyph_detection_main.py". Bring various glyphs in front of the camera and try to see the results. The glyph detection part is very sensitive to lighting conditions and hence you may want to ensure proper lights on the glyph.

#### 2D AR - Superpose glyphs with an image
Run "detection.py"

# Results 
We were able to succesfully implement 3D object overaly over glyphs. One needs to be patient while working with VTK ( Python wrapper for OpenGL).
![cube_and_totoro](https://user-images.githubusercontent.com/25552500/39655961-5aaeee86-501a-11e8-8877-7c895729464a.png)

# References
see the file Glyph-References.pdf
