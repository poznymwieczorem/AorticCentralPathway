import vtk
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
from vtkmodules.util.numpy_support import vtk_to_numpy
import matplotlib
matplotlib.use('TkAgg')  # lub 'Qt5Agg' jeśli masz PyQt5/6
import matplotlib.pyplot as plt

def vtp_to_binary_volume(vtp_path, spacing=(1.0, 1.0, 1.0)):
    # Wczytaj plik VTP
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()

    # Określ rozmiar objętości na podstawie bounding boxa
    bounds = polydata.GetBounds()  # xmin, xmax, ymin, ymax, zmin, zmax
    dims = [
        int((bounds[1] - bounds[0]) / spacing[0]) + 1,
        int((bounds[3] - bounds[2]) / spacing[1]) + 1,
        int((bounds[5] - bounds[4]) / spacing[2]) + 1
    ]

    # Przekształć PolyData do voxelowej siatki (binary image)
    white_image = vtk.vtkImageData()
    white_image.SetSpacing(spacing)
    white_image.SetDimensions(dims)
    white_image.SetExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
    white_image.SetOrigin(bounds[0], bounds[2], bounds[4])
    white_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # Wypełnij białym tłem (1)
    white_image.GetPointData().GetScalars().Fill(1)

    # Tworzenie obiektu do rasteryzacji (pola binarne)
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin(white_image.GetOrigin())
    pol2stenc.SetOutputSpacing(white_image.GetSpacing())
    pol2stenc.SetOutputWholeExtent(white_image.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(white_image)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()

    # Konwersja do numpy
    vtk_image = imgstenc.GetOutput()
    dims = vtk_image.GetDimensions()
    scalars = vtk_image.GetPointData().GetScalars()
    np_array = vtk_to_numpy(scalars)
    np_array = np_array.reshape(dims[2], dims[1], dims[0])  # Uwaga: kolejność osi Z, Y, X /// Y, X, Z

    return np_array, vtk_image.GetOrigin(), vtk_image.GetSpacing()

def crop_to_colon(binary_volume):
    coords = np.argwhere(binary_volume)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1
    cropped = binary_volume[min_coords[0]:max_coords[0],
                             min_coords[1]:max_coords[1],
                             min_coords[2]:max_coords[2]]
    return cropped, min_coords


def compute_dbf(cropped):
    inv = np.logical_not(cropped)
    return distance_transform_edt(cropped) - distance_transform_edt(inv)


def compute_gvf(dbf):
    gy, gx, gz = np.gradient(dbf)
    norm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-5
    return np.stack((gx / norm, gy / norm, gz / norm), axis=-1)


def flag_nonuniform_regions(gvf, threshold=0.2):
    grad_mag = np.linalg.norm(gvf, axis=-1)
    uniformity = gaussian_filter(grad_mag, sigma=1)
    return (uniformity < threshold)


def connect_flagged_voxels(flags):
    return skeletonize(flags.astype(np.uint8))


def compute_daf(flags):
    return distance_transform_edt(np.logical_not(flags))


def compute_pdef(daf, end_voxel):
    from queue import PriorityQueue
    shape = daf.shape
    pdef = np.full(shape, np.inf)
    pdef[end_voxel] = 0
    visited = np.zeros(shape, dtype=bool)
    pq = PriorityQueue()
    pq.put((0, end_voxel))

    while not pq.empty():
        dist, current = pq.get()
        if visited[current]: continue
        visited[current] = True
        x, y, z = current
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz == 0: continue
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                        new_dist = dist + daf[nx, ny, nz]
                        if new_dist < pdef[nx, ny, nz]:
                            pdef[nx, ny, nz] = new_dist
                            pq.put((new_dist, (nx, ny, nz)))
    return pdef


def estimate_endpoints(binary_volume):
    coords = np.argwhere(binary_volume)
    ranges = coords.max(axis=0) - coords.min(axis=0)
    main_axis = np.argmax(ranges)
    sorted_coords = coords[np.argsort(coords[:, main_axis])]
    return tuple(sorted_coords[0]), tuple(sorted_coords[-1])


def extract_path(pdef, start_voxel, end_voxel):
    path, _ = route_through_array(pdef, start_voxel, end_voxel, fully_connected=True)
    return np.array(path)


def smooth_centerline(path, sigma=1):
    return gaussian_filter(path.astype(np.float32), sigma=[0, 0])


def save_path_as_vtp(path_coords, origin, spacing, output_path):
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    for i, voxel in enumerate(path_coords):
        real_pos = voxel * spacing + origin
        points.InsertNextPoint(real_pos.tolist())
        if i > 0:
            lines.InsertNextCell(2)
            lines.InsertCellPoint(i - 1)
            lines.InsertCellPoint(i)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.Write()


def visualize(binary_volume, origin, spacing, path=None):
    importer = vtk.vtkImageImport()
    data_string = binary_volume.astype(np.uint8).tobytes()
    importer.CopyImportVoidPointer(data_string, len(data_string))
    importer.SetDataScalarTypeToUnsignedChar()
    importer.SetNumberOfScalarComponents(1)
    dz, dy, dx = binary_volume.shape
    importer.SetDataExtent(0, dx - 1, 0, dy - 1, 0, dz - 1)
    importer.SetWholeExtent(0, dx - 1, 0, dy - 1, 0, dz - 1)
    importer.SetDataSpacing(spacing)
    importer.SetDataOrigin(origin)
    importer.Update()

    opacity = vtk.vtkPiecewiseFunction()
    opacity.AddPoint(0, 0.0)
    opacity.AddPoint(1, 0.2)

    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(0, 0.0, 0.0, 0.0)
    color.AddRGBPoint(1, 1.0, 1.0, 1.0)

    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color)
    volume_property.SetScalarOpacity(opacity)
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()

    volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
    volume_mapper.SetInputConnection(importer.GetOutputPort())

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0, 0, 0)

    if path is not None:
        path_points = vtk.vtkPoints()
        path_lines = vtk.vtkCellArray()

        for i, voxel in enumerate(path):
            real_pos = voxel * spacing + origin
            path_points.InsertNextPoint(real_pos.tolist())
            if i > 0:
                path_lines.InsertNextCell(2)
                path_lines.InsertCellPoint(i - 1)
                path_lines.InsertCellPoint(i)

        path_poly = vtk.vtkPolyData()
        path_poly.SetPoints(path_points)
        path_poly.SetLines(path_lines)

        path_mapper = vtk.vtkPolyDataMapper()
        path_mapper.SetInputData(path_poly)

        path_actor = vtk.vtkActor()
        path_actor.SetMapper(path_mapper)
        path_actor.GetProperty().SetColor(1, 0, 0)
        path_actor.GetProperty().SetLineWidth(3)

        renderer.AddActor(path_actor)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    render_window.Render()
    interactor.Start()


def show_slices(volume):
    """
    Pokazuje środkowe przekroje 3D macierzy `volume` w osiach Z, Y, X.

    Parameters:
        volume (np.ndarray): 3D macierz binarna (Z, Y, X)
    """
    z, y, x = volume.shape

    # Wyznacz środkowe indeksy przekrojów
    slice_z = volume[z // 2, :, :]
    slice_y = volume[:, y // 2, :]
    slice_x = volume[:, :, x // 2]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(slice_z, cmap='gray')
    axes[0].set_title('Przekrój Z (XY)')
    axes[0].axis('off')

    axes[1].imshow(slice_y, cmap='gray')
    axes[1].set_title('Przekrój Y (XZ)')
    axes[1].axis('off')

    axes[2].imshow(slice_x, cmap='gray')
    axes[2].set_title('Przekrój X (YZ)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    input_vtp = r"D:\TOM_project_\0070_H_AO_H\0070_H_AO_H\Models\0183_1002_aorta.vtp"  # zmień nazwę pliku wejściowego
    output_path = 'centerline.vtp'

    binary, origin, spacing = vtp_to_binary_volume(input_vtp, spacing=(0.01, 0.01, 0.01))
    print(binary.shape)
    show_slices(binary)
    cropped, offset = crop_to_colon(binary)
    dbf = compute_dbf(cropped)
    gvf = compute_gvf(dbf)
    flags = flag_nonuniform_regions(gvf)
    connected = connect_flagged_voxels(flags)
    daf = compute_daf(connected)
    start_voxel, end_voxel = estimate_endpoints(cropped)
    pdef = compute_pdef(daf, end_voxel)
    path = extract_path(pdef, start_voxel, end_voxel)
    smoothed = smooth_centerline(path)
    save_path_as_vtp(smoothed, origin=np.array(origin) + offset * spacing, spacing=spacing, output_path=output_path)
    visualize(cropped, origin=np.array(origin) + offset * spacing, spacing=spacing, path=smoothed)
