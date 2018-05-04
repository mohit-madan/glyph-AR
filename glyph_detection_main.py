#!/usr/bin/env python
"""
based on:
 - http://www.vtk.org/Wiki/VTK/Examples/Cxx/Images/BackgroundImage

"""
from __future__ import print_function
from vtk import *
import cv2
from detection_3D import capture
import numpy as np
from order_pts import order_pts

ACTIVATE_TOTORO = False


def draw(img, corner, imgpts):
    corner = tuple(corner)
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def draw_cage(img, corner, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (10, 120, 5), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 1)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 1)

    return img


def get_vectors(image, points, coord, mtx, dist):
    # order points
    points = order_pts(points)

    # set up criteria, image, points and axis
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imgp = np.array(points, dtype="float32")

    objp = np.array(coord, dtype="float32")

    # calculate rotation and translation vectors
    imgp = cv2.cornerSubPix(gray, imgp, (11, 11), (-1, -1), criteria)
    _, rvecs, tvecs = cv2.solvePnP(objp, imgp, mtx, dist)

    return rvecs, tvecs


def main(argv):
    # set window size
    width = 640
    height = 480

    # load camera calibration data
    with np.load('camcalib.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
    # set focal length
    focalLengthY = mtx[1][1]
    viewAngle = 2 * np.arctan(height / 2 / focalLengthY) * 180 / np.pi

    # read first frame
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cv2.imwrite('webcam.jpg', frame)

    # Create background with webcam input
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


    # Object 1: Cube
    # Read .obj
    file_name = 'data/3d_tree/3d_tree.obj'
    reader = vtk.vtkOBJReader()
    reader.SetFileName(file_name)
    reader.Update()
    poly_data = reader.GetOutput()

    modelMapper = vtk.vtkPolyDataMapper()
    modelMapper.SetInputData(poly_data)

    CubeActor = vtk.vtkActor()
    CubeActor.SetMapper(modelMapper)
    cube_coord = [[-1., -1., 0.], [1., -1., 0.],
                  [1., 1., 0.], [-1., 1., 0.]]

    # Object 2: Totoro
    # Read .obj
    if ACTIVATE_TOTORO:
        file_name = 'data/totoro_target.obj'
        totoro_coord = [[-0.5, 0., -0.5], [0.5, 0., -0.5],
                  [0.5, 0., 0.5], [-0.5, 0., 0.5]]
    else:
        file_name = 'data/3d_tree/3d_tree.obj'
        totoro_coord = [[-2., -2., 0.], [2., -2., 0.],
                  [2., 2., 0.], [-2., 2., 0.]]

    reader = vtk.vtkOBJReader()
    reader.SetFileName(file_name)
    reader.Update()
    poly_data = reader.GetOutput()

    totoroMapper = vtk.vtkPolyDataMapper()
    totoroMapper.SetInputData(poly_data)

    totoroActor = vtk.vtkActor()
    totoroActor.SetMapper(totoroMapper)


    # Object 3: cage_axis to be displayed on glyph
    cage_axis = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                            [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])
    cage_coord = [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.],[0., 1., 0.]]



    # create renderer to display objects
    cube_renderer = vtkRenderer()
    totoro_renderer = vtkRenderer()

    # create render window to combine all renderer
    render_window = vtkRenderWindow()
    render_window.SetSize(width, height)

    # Set up the render window and renderer such that there is
    # a background layer and a foreground layer
    background_renderer.SetLayer(0)
    background_renderer.InteractiveOff()
    cube_renderer.SetLayer(1)
    totoro_renderer.SetLayer(2)
    render_window.SetNumberOfLayers(3)
    render_window.AddRenderer(background_renderer)
    render_window.AddRenderer(cube_renderer)
    render_window.AddRenderer(totoro_renderer)

    render_window_interactor = vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add actors to the renderer
    background_renderer.AddActor(image_actor)
    cube_renderer.AddActor(CubeActor)
    totoro_renderer.AddActor(totoroActor)

    # Render once to figure out where the background camera will be
    render_window.Render()
    # Set up the background camera to fill the renderer with the image
    origin = image_data.GetOrigin()
    spacing = image_data.GetSpacing()
    extent = image_data.GetExtent()

    camera = background_renderer.GetActiveCamera()
    camera.ParallelProjectionOn()
    xc = origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0]
    yc = origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1]
    # xd = (extent[1] - extent[0] + 1) * spacing[0]
    yd = (extent[3] - extent[2] + 1) * spacing[1]
    d = camera.GetDistance()
    camera.SetParallelScale(0.5 * yd)
    # camera.SetFocalPoint(xc, yc, 0.0)
    # camera.SetPosition(xc, yc, d)

    # setup object camera according to webcam intrinsics
    cube_camera = cube_renderer.GetActiveCamera()
    cube_camera.SetViewAngle(viewAngle)
    totoro_camera = totoro_renderer.GetActiveCamera()
    totoro_camera.SetViewAngle(viewAngle)


    # Render again to set the correct view
    render_window.Render()

    # start loop
    count0 = 0
    count1 = 0
    count2 = 0
    while True:
        ret, frame = cam.read()

        idx, approx = capture(frame)

        if idx == 0:
            cube_renderer.AddActor(CubeActor)
            (tl, tr, br, bl) = order_pts(approx)
            rvecs, tvecs = get_vectors(frame, approx, cube_coord, mtx, dist)
            rmat, _ = cv2.Rodrigues(rvecs)

            # method 2
            # tweaked version of: https://stackoverflow.com/questions/25539898/how-to-apply-the-camera-pose-transformation-computed-using-epnp-to-the-vtk-camer
            rmat[1][0] *= -1
            rmat[1][1] *= -1
            rmat[1][2] *= -1
            rmat[2][0] *= -1
            rmat[2][1] *= -1
            rmat[2][2] *= -1
            tvec = tvecs.copy()
            tvec[1] *= -1
            tvec[2] *= -1

            viewPlaneNormal = (rmat[2][0].copy(), rmat[2][1].copy(), rmat[2][2].copy())
            rmatINV = rmat.copy()
            rmatINV = rmatINV.transpose()

            translation = rmatINV.dot(tvec)
            translation *= -1

            # set camera
            cube_camera.SetPosition(translation[0], translation[1], translation[2])

            cube_camera.SetFocalPoint(translation[0][0] - viewPlaneNormal[0], translation[1][0] - viewPlaneNormal[1],
                                     translation[2][0] - viewPlaneNormal[2])

            cube_camera.SetViewUp(rmat[1][0], rmat[1][1], rmat[1][2])

            cube_renderer.ResetCameraClippingRange()

            # use counter to stabelize projection
            count0 = 10
        # keep cube active to make up for glyph-recognition
        count0 -= 1
        if count0 <= 0:
            cube_renderer.RemoveActor(CubeActor)


        if idx == 1:
            totoro_renderer.AddActor(totoroActor)
            (tl, tr, br, bl) = order_pts(approx)
            rvecs, tvecs = get_vectors(frame, approx, totoro_coord, mtx, dist)
            rmat, _ = cv2.Rodrigues(rvecs)

            # method 2
            rmat[1][0] *= -1
            rmat[1][1] *= -1
            rmat[1][2] *= -1
            rmat[2][0] *= -1
            rmat[2][1] *= -1
            rmat[2][2] *= -1
            tvec = tvecs.copy()
            tvec[1] *= -1
            tvec[2] *= -1

            viewPlaneNormal = (rmat[2][0].copy(), rmat[2][1].copy(), rmat[2][2].copy())
            rmatINV = rmat.copy()
            rmatINV = rmatINV.transpose()

            translation = rmatINV.dot(tvec)
            translation *= -1

            # defines depth position of totoro
            totoro_camera.SetPosition(translation[0], translation[1], translation[2])

            totoro_camera.SetFocalPoint(translation[0][0] - viewPlaneNormal[0], translation[1][0] - viewPlaneNormal[1],
                                     translation[2][0] - viewPlaneNormal[2])

            totoro_camera.SetViewUp(rmat[1][0], rmat[1][1], rmat[1][2])

            totoro_renderer.ResetCameraClippingRange()

            count2 = 10
        # keep totoro active to make up for glyph-recognition
        count2 -= 1
        if count2 <= 0:
            totoro_renderer.RemoveActor(totoroActor)



        # Action Nr. 3: Draw cube on image
        if idx == 1:
            (tl, tr, br, bl) = order_pts(approx)
            rvecs, tvecs = get_vectors(frame, approx, cage_coord, mtx, dist)
            imgpts, jac = cv2.projectPoints(cage_axis, rvecs, tvecs, mtx, dist)
            count1 = 10;

        if count1 > 0:
            frame = draw_cage(frame, tl, imgpts)
            count1 -= 1


        cv2.imwrite('webcam.jpg', frame)

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
