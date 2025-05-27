import vtk
import numpy as np
from skimage.morphology import skeletonize_3d
from vtk.util import numpy_support


def read_vtp(file_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()


def convert_polydata_to_binary_image(polydata, spacing=1.0):
    bounds = polydata.GetBounds()
    dims = [int((bounds[i * 2 + 1] - bounds[i * 2]) / spacing) + 1 for i in range(3)]
    origin = [bounds[0], bounds[2], bounds[4]]

    white_image = vtk.vtkImageData()
    white_image.SetSpacing(spacing, spacing, spacing)
    white_image.SetDimensions(dims)
    white_image.SetExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
    white_image.SetOrigin(origin)
    white_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(spacing, spacing, spacing)
    pol2stenc.SetOutputWholeExtent(white_image.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(white_image)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()

    return imgstenc.GetOutput()


def skeletonize_image(image_data):
    dims = image_data.GetDimensions()
    scalars = image_data.GetPointData().GetScalars()
    voxel_data = numpy_support.vtk_to_numpy(scalars).reshape(dims[::-1])
    binary_volume = voxel_data > 0
    skeleton = skeletonize_3d(binary_volume)
    return skeleton


def skeleton_to_polyline(skeleton, spacing, origin):
    points = np.argwhere(skeleton)
    vtk_points = vtk.vtkPoints()
    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(len(points))

    for i, point in enumerate(points):
        world_point = point * spacing + origin
        vtk_points.InsertNextPoint(world_point)
        polyline.GetPointIds().SetId(i, i)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(cells)

    return polydata


def display_geometries(original_polydata, optional_path=None, computed_centerline=None):
    renderers = []

    def create_actor(polydata, color):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        return actor

    # Scene 1: Original aorta
    renderer1 = vtk.vtkRenderer()
    renderer1.AddActor(create_actor(original_polydata, (1, 1, 1)))
    renderers.append(renderer1)

    # Scene 2: Aorta + optional path
    renderer2 = vtk.vtkRenderer()
    renderer2.AddActor(create_actor(original_polydata, (0.9, 0.9, 0.9)))
    if optional_path:
        renderer2.AddActor(create_actor(optional_path, (1, 0, 0)))
    renderers.append(renderer2)

    # Scene 3: Aorta + computed centerline
    renderer3 = vtk.vtkRenderer()
    renderer3.AddActor(create_actor(original_polydata, (0.9, 0.9, 0.9)))
    if computed_centerline:
        renderer3.AddActor(create_actor(computed_centerline, (0, 1, 0)))
    renderers.append(renderer3)

    # Setup render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1200, 400)
    for i, ren in enumerate(renderers):
        viewport = [i / 3.0, 0.0, (i + 1) / 3.0, 1.0]
        ren.SetViewport(*viewport)
        render_window.AddRenderer(ren)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(render_window)
    render_window.Render()
    iren.Start()


# Main example execution (disabled here to avoid running without user input)
file_path = r"D:\TOM_project_\0143_H_AO_H\Models\KDR34_aorta.vtp"
polydata = read_vtp(file_path)
image_data = convert_polydata_to_binary_image(polydata)
skeleton = skeletonize_image(image_data)
centerline = skeleton_to_polyline(skeleton, spacing=1.0, origin=[polydata.GetBounds()[0], polydata.GetBounds()[2], polydata.GetBounds()[4]])
display_geometries(polydata, optional_path=None, computed_centerline=centerline)

