import vtk
import numpy as np
from scipy.spatial import distance
from scipy.ndimage import distance_transform_edt

def read_vtp(file_path):
    """Wczytuje plik .vtp i zwraca vtkPolyData"""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def polydata_to_numpy(polydata):
    """Konwertuje vtkPolyData do chmury punktów numpy"""
    points = polydata.GetPoints()
    np_points = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
    return np_points

def get_centerline_skeleton(polydata):
    """
    Używa algorytmu skeletyzacji 3D do wyznaczenia ścieżki centralnej z siatki
    Wymaga, by polydata reprezentowała zamkniętą strukturę (np. siatkę aorty)
    """
    # Zamiana siatki na obraz binarny (3D volume)
    bounds = polydata.GetBounds()
    spacing = 1.0  # rozdzielczość voxeli
    dims = [int((bounds[i * 2 + 1] - bounds[i * 2]) / spacing) + 1 for i in range(3)]

    white_image = vtk.vtkImageData()
    white_image.SetSpacing(spacing, spacing, spacing)
    white_image.SetDimensions(dims)
    white_image.SetExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)

    origin = [bounds[0], bounds[2], bounds[4]]
    white_image.SetOrigin(origin)

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

    # Konwersja do numpy
    img_data = imgstenc.GetOutput()
    dims = img_data.GetDimensions()
    scalars = img_data.GetPointData().GetScalars()
    voxel_data = vtk.util.numpy_support.vtk_to_numpy(scalars).reshape(dims[::-1])

    # Skeletyzacja na danych binarnych
    from skimage.morphology import skeletonize_3d
    binary_volume = voxel_data > 0
    skeleton = skeletonize_3d(binary_volume)

    # Wyodrębnienie ścieżki jako chmury punktów
    centerline_points = np.argwhere(skeleton)
    centerline_world = centerline_points * spacing + np.array(origin)

    return centerline_world

# Przykład użycia
if __name__ == "__main__":
    vtp_path = r"D:\TOM_project_\0143_H_AO_H\Models\KDR34_aorta.vtp"
    polydata = read_vtp(vtp_path)
    centerline = get_centerline_skeleton(polydata)

    print(f"Wyznaczono {len(centerline)} punktów ścieżki centralnej.")