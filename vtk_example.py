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



def main(argv):
    cam = cv2.VideoCapture(0)

    #  Verify input arguments
    if True:
        # Read the image
        path = 'webcam0.jpg'
        jpeg_reader = vtkJPEGReader()
        if not jpeg_reader.CanReadFile(path):
            print("Error reading file %s" % path)
            return

        jpeg_reader.SetFileName(path)
        jpeg_reader.Update()
        image_data = jpeg_reader.GetOutput()
    else:
        canvas_source = vtkImageCanvasSource2D()
        canvas_source.SetExtent(0, 100, 0, 100, 0, 0)
        canvas_source.SetScalarTypeToUnsignedChar()
        canvas_source.SetNumberOfScalarComponents(3)
        canvas_source.SetDrawColor(127, 127, 100)
        canvas_source.FillBox(0, 100, 0, 100)
        canvas_source.SetDrawColor(100, 255, 255)
        canvas_source.FillTriangle(10, 10, 25, 10, 25, 25)
        canvas_source.SetDrawColor(255, 100, 255)
        canvas_source.FillTube(75, 75, 0, 75, 5.0)
        canvas_source.Update()
        image_data = canvas_source.GetOutput()

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

    modelActor = vtk.vtkActor()
    modelActor.SetMapper(modelMapper)
    modelActor.SetPosition(0, 0, -100)
    # modelActor.RotateX(30)
    # modelActor.RotateY(70)
    # modelActor.RotateZ(100)


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
    scene_renderer.AddActor(modelActor)
    background_renderer.AddActor(image_actor)

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
    camera.SetFocalPoint(xc, yc, 0.0)
    camera.SetPosition(xc, yc, d)



    # Render again to set the correct view
    render_window.Render()

    while True:
        ret, frame = cam.read()
        cv2.imwrite('webcam.jpg',frame)

        idx, approx = capture(frame)
        if idx == 0:
            print(idx, approx)
            # scene_renderer.AddActor(modelActor)
            modelActor.AddPosition(0.1, 0, 0)
        elif idx == 1:
            print(idx, approx)
            # scene_renderer.AddActor(modelActor)
            modelActor.AddPosition(-0.1, 0, 0)


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
