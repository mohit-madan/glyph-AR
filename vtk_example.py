#!/usr/bin/env python
"""
based on:
 - http://www.vtk.org/Wiki/VTK/Examples/Cxx/Images/BackgroundImage

tested with:
 - VTK 6.3, Python 2.7
 - VTK 7.0, Python 3.5

>>> python background_image.py image_filename.jpg
"""
from __future__ import print_function
import sys
# from vtk import (
#     vtkJPEGReader, vtkImageCanvasSource2D, vtkImageActor, vtkPolyDataMapper,
#     vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkSuperquadricSource,
#     vtkActor, VTK_MAJOR_VERSION
# )
from vtk import *
import cv2
from detection_3D import capture
import numpy as np
from order_pts import order_pts
def draw(img, corner, imgpts):
    corner = tuple(corner)
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

def draw_cage(img, corner, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),-3)

    return img


def get_vectors(image, points, mtx, dist):

    # order points
    points = order_pts(points)

    # set up criteria, image, points and axis
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imgp = np.array(points, dtype="float32")

    objp = np.array([[0.,0.,0.],[1.,0.,0.],
                        [1.,1.,0.],[0.,1.,0.]], dtype="float32")

    # calculate rotation and translation vectors
    imgp = cv2.cornerSubPix(gray,imgp,(11,11),(-1,-1),criteria)
    _, rvecs, tvecs = cv2.solvePnP(objp, imgp, mtx, dist)

    return rvecs, tvecs


def main(argv):

    # axis to be displayed on glyph
    axis = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                   [0,0,-1],[0,1,-1],[1,1,-1],[1,0,-1] ])    # load calibration data
    with np.load('camcalib.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cv2.imwrite('webcam.jpg', frame)

    #  Verify input arguments
    # Read the image
    path = 'webcam.jpg'
    jpeg_reader = vtkJPEGReader()
    if not jpeg_reader.CanReadFile(path):
        print("Error reading file %s" % path)
        return

    jpeg_reader.SetFileName(path)
    jpeg_reader.Update()
    image_data = jpeg_reader.GetOutput()


    # Create an image actor to display the image
    image_actor = vtkImageActor()

    if VTK_MAJOR_VERSION <= 5:
        image_actor.SetInput(image_data)
    else:
        image_actor.SetInputData(image_data)

    # Create a renderer to display the image in the background
    background_renderer = vtkRenderer()

    # Read .obj
    file_name = 'data/3d_tree/3d_tree.obj'
    reader = vtk.vtkOBJReader()
    reader.SetFileName(file_name)
    reader.Update()
    poly_data = reader.GetOutput()

    # make transfor for translate
    # aLabelTransform = vtk.vtkTransform()
    # aLabelTransform.Identity()
    # aLabelTransform.Translate(-0.2, 0, 1.25)
    #
    # labelTransform = vtk.vtkTransformPolyDataFilter()
    # labelTransform.SetTransform(aLabelTransform)
    # labelTransform.SetInputData(poly_data)

    modelMapper = vtk.vtkPolyDataMapper()
    modelMapper.SetInputData(poly_data)

    CubeActor = vtk.vtkActor()
    CubeActor.SetMapper(modelMapper)
    # CubeActor.RotateX(30)
    # CubeActor.RotateY(70)
    # CubeActor.RotateZ(100)


    # Create a superquadric
    superquadric_source = vtkSuperquadricSource()
    superquadric_source.SetPhiRoundness(1.1)
    superquadric_source.SetThetaRoundness(.2)


    # Create a mapper and actor
    superquadric_mapper = vtk.vtkPolyDataMapper()
    superquadric_mapper.SetInputConnection(superquadric_source.GetOutputPort())

    superquadric_actor = vtkActor()
    superquadric_actor.SetMapper(superquadric_mapper)
    superquadric_actor.RotateX(30)
    superquadric_actor.RotateY(70)
    superquadric_actor.RotateZ(100)
    # superquadric_actor.AddPosition(5, 0, 0)


    scene_renderer = vtkRenderer()

    render_window = vtkRenderWindow()
    render_window.SetSize(640, 480)

    # Set up the render window and renderers such that there is
    # a background layer and a foreground layer
    background_renderer.SetLayer(0)
    background_renderer.InteractiveOff()
    scene_renderer.SetLayer(1)
    render_window.SetNumberOfLayers(2)
    render_window.AddRenderer(background_renderer)
    render_window.AddRenderer(scene_renderer)

    render_window_interactor = vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add actors to the renderers
    # scene_renderer.AddActor(superquadric_actor)
    background_renderer.AddActor(image_actor)
    # scene_renderer.AddActor(CubeActor)

    # Render once to figure out where the background camera will be
    render_window.Render()


    # Set up the background camera to fill the renderer with the image
    origin = image_data.GetOrigin()
    spacing = image_data.GetSpacing()
    extent = image_data.GetExtent()

    camera = background_renderer.GetActiveCamera()
    camera.ParallelProjectionOn()

    scn_camera = scene_renderer.GetActiveCamera()
    # scn_camera.ParallelProjectionOn()

    xc = origin[0] + 0.5*(extent[0] + extent[1]) * spacing[0]
    yc = origin[1] + 0.5*(extent[2] + extent[3]) * spacing[1]
    # xd = (extent[1] - extent[0] + 1) * spacing[0]
    yd = (extent[3] - extent[2] + 1) * spacing[1]
    d = camera.GetDistance()
    camera.SetParallelScale(0.5 * yd)
    # camera.SetFocalPoint(xc, yc, 0.0)
    # camera.SetPosition(xc, yc, d)





    # Render again to set the correct view
    render_window.Render()

    count = 0

    while True:
        ret, frame = cam.read()

        idx, approx = capture(frame)
        if idx == 0 or idx == 1:
            (tl, tr, br, bl) = order_pts(approx)
            rvecs, tvecs = get_vectors(frame, approx, mtx, dist)
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            count = 10;

        if count > 0:
            frame = draw_cage(frame,tl, imgpts)
            count -= 1;

        cv2.imwrite('webcam.jpg', frame)

        # if idx == 0:
        #     print(idx, approx)
        #     # scene_renderer.AddActor(CubeActor)
        #     CubeActor.AddPosition(0.1, 0, 0)
        # elif idx == 1:
        #     print(idx, approx)
        #     # scene_renderer.AddActor(CubeActor)
        #     CubeActor.AddPosition(-0.1, 0, 0)


        # Read the image to background
        path = 'webcam.jpg'
        jpeg_reader = vtkJPEGReader()
        if not jpeg_reader.CanReadFile(path):
            print("Error reading file %s" % path)
            return

        jpeg_reader.SetFileName(path)
        jpeg_reader.Update()
        image_data = jpeg_reader.GetOutput()
        if VTK_MAJOR_VERSION <= 5:
            image_actor.SetInput(image_data)
        else:
            image_actor.SetInputData(image_data)

        render_window.Render()


    # Interact with the window / keeps window open otherwise closes after loop is finished
    render_window_interactor.Start()


if __name__ == '__main__':
    main(sys.argv)
