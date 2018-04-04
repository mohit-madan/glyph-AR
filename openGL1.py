from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
from PIL import Image
import numpy as np
from objloader import *
from pattern_recognition import pattern_recognition
from order_pts import check_if_rect, order_pts
from extractMatrix import extractMatrix


def get_vectors(image, points):

    # order points
    points = order_pts(points)

    # load calibration data
    with np.load('webcam_calibration_ouput.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

    # set up criteria, image, points and axis
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imgp = np.array(points, dtype="float32")

    objp = np.array([[0.,0.,0.],[1.,0.,0.],
                        [1.,1.,0.],[0.,1.,0.]], dtype="float32")

    # calculate rotation and translation vectors
    cv2.cornerSubPix(gray,imgp,(11,11),(-1,-1),criteria)
    rvecs, tvecs, _ = cv2.solvePnPRansac(objp, imgp, mtx, dist)

    return rvecs, tvecs


class OpenGLGlyphs:

    # constants
    INVERSE_MATRIX = np.array([[ 1.0, 1.0, 1.0, 1.0],
                               [-1.0,-1.0,-1.0,-1.0],
                               [-1.0,-1.0,-1.0,-1.0],
                               [ 1.0, 1.0, 1.0, 1.0]])

    def __init__(self):
        # initialise webcam and start thread
        self.webcam = cv2.VideoCapture(0)

        # initialise shapes
        self.cone = None
        self.sphere = None

        # initialise texture
        self.texture_background = None

    def _init_gl(self, Width, Height):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(33.7, 1.3, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        # assign shapes
        self.cone = OBJ('data/3d_tree/3d_tree.obj')
        self.sphere = OBJ('data/3d_tree/3d_tree.obj')

        # assign texture
        glEnable(GL_TEXTURE_2D)
        self.texture_background = glGenTextures(1)

    def _draw_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # get image from webcam
        ret, image = self.webcam.read()

        # convert image to OpenGL texture format
        bg_image = cv2.flip(image, 0)
        bg_image = Image.fromarray(bg_image)
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)

        # create background texture
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)

        # draw background
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glPushMatrix()
        glTranslatef(0.0,0.0,-10.0)
        self._draw_background()
        glPopMatrix()

        # handle glyphs
        self._handle_glyphs(image)

        glutSwapBuffers()

    def _handle_glyphs(self, image):

        # attempt to detect glyphs
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)  # applying histogram equalisation

        gray = cv2.GaussianBlur(gray, (5, 5), 1)  # gaussian blur-to smoothen out random edges
        gray_edge = cv2.Canny(gray, 100, 200)  # applying canny edge detection
        # cv2.imshow("jf", gray_edge)
        im2, contours, _ = cv2.findContours(gray_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # finding contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # sorting contours in reverse order - why? don't know
        # approximating contours

        for cnt in contours:
            # We will find if each of the detected contour is of quad shape, then we will do the perspective
            # transform of the image to get the quad in top down view and then the glyph detection algo will proceed
            epsilon = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01*epsilon, True)  # with greater percentage a large set is coming
            cv2.drawContours(image, cnt, -1, (0, 255, 0), 3)
            cv2.drawContours(image, approx, -1, (0, 0, 255), 3)
            # cv2.imshow('frame', image)
            vert_num = len(approx)
            if vert_num == 4:
                approx = approx.reshape(4, 2)
                (tl, tr, br, bl) = order_pts(approx)
                valid = check_if_rect(approx)
                # valid = qFalse

                if valid:
                    # i += 1
                    # print(i)
                    # print(approx)
                    warped_img, H = extractMatrix(gray, approx)
                    cv2.imshow("original", gray)
                    cv2.imshow("transformed", warped_img)

                    idx = pattern_recognition(warped_img)

                    if idx != None:

                        rvecs, tvecs = get_vectors(image, approx)

                        # build view matrix
                        rmtx = cv2.Rodrigues(rvecs)[0]

                        view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],tvecs[0]],
                                                [rmtx[1][0],rmtx[1][1],rmtx[1][2],tvecs[1]],
                                                [rmtx[2][0],rmtx[2][1],rmtx[2][2],tvecs[2]],
                                                [0.0       ,0.0       ,0.0       ,1.0    ]])

                        view_matrix = view_matrix * self.INVERSE_MATRIX

                        view_matrix = np.transpose(view_matrix)

                        # load view matrix and draw shape
                        glPushMatrix()
                        glLoadMatrixd(view_matrix)

                        if idx == 0:
                            glCallList(self.cone.gl_list)
                        elif idx == 1:
                            glCallList(self.sphere.gl_list)

                        glPopMatrix()

    def _draw_background(self):
        # draw background
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 4.0,  3.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-4.0,  3.0, 0.0)
        glEnd( )

    def main(self):
        # setup and run OpenGL
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(640, 480)
        glutInitWindowPosition(800, 400)
        self.window_id = glutCreateWindow("OpenGL Glyphs")
        glutDisplayFunc(self._draw_scene)
        glutIdleFunc(self._draw_scene)
        self._init_gl(640, 480)
        glutMainLoop()

# run an instance of OpenGL Glyphs
openGLGlyphs = OpenGLGlyphs()
openGLGlyphs.main()
