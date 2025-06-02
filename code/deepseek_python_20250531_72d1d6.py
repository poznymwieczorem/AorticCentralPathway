import vtk
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
from vtkmodules.util.numpy_support import vtk_to_numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def vtp_to_binary_volume(vtp_path, spacing=(1.0, 1.0, 1.0)):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()

    bounds = polydata.GetBounds()
    dims = [
        int((bounds[1] - bounds[0]) / spacing[0]) + 1,
        int((bounds[3] - bounds[2]) / spacing[1]) + 1,
        int((bounds[5] - bounds[4]) / spacing[2]) + 1
    ]

    white_image = vtk.vtkImageData()
    white_image.SetSpacing(spacing)
    white_image.SetDimensions(dims)
    white_image.SetExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
    white_image.SetOrigin(bounds[0], bounds[2], bounds[4])
    white_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    white_image.GetPointData().GetScalars().Fill(1)

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

    vtk_image = imgstenc.GetOutput()
    dims = vtk_image.GetDimensions()
    scalars = vtk_image.GetPointData().GetScalars()
    np_array = vtk_to_numpy(scalars)
    np_array = np_array.reshape(dims[2], dims[1], dims[0])  # Z, Y, X

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
    norm = np.sqrt(gx**2 + gy**2 + gz**2) + 1e-5
    return np.stack((gx/norm, gy/norm, gz/norm), axis=-1)

def flag_nonuniform_regions(gvf, threshold=0.5):
    grad_mag = np.linalg.norm(gvf, axis=-1)
    uniformity = gaussian_filter(grad_mag, sigma=2)
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
    """
    Ulepszona metoda znajdowania punktów końcowych aorty:
    1. Oblicza szkielet objętości
    2. Identyfikuje punkty końcowe (z tylko jednym sąsiadem)
    3. Wybiera najbardziej oddaloną parę jako końce aorty
    4. Weryfikuje, czy punkty leżą na brzegu objętości
    """
    # Oblicz szkielet
    skeleton = skeletonize(binary_volume)

    # Znajdź wszystkie punkty końcowe szkieletu
    endpoints = []
    coords = np.argwhere(skeleton)

    for voxel in coords:
        x, y, z = voxel
        # Policz aktywnych sąsiadów (26-connectivity)
        neighbors = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (0 <= nx < skeleton.shape[0] and
                            0 <= ny < skeleton.shape[1] and
                            0 <= nz < skeleton.shape[2] and
                            skeleton[nx, ny, nz]):
                        neighbors += 1
        if neighbors == 1:  # Punkt końcowy ma tylko 1 sąsiada
            endpoints.append((x, y, z))

    # Jeśli znaleziono mniej niż 2 punkty końcowe, użyj alternatywnej metody
    if len(endpoints) < 2:
        return find_endpoints_by_distance(binary_volume)

    # Znajdź najbardziej oddaloną parę punktów końcowych
    max_dist = -1
    best_pair = None
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            dist = np.linalg.norm(np.array(endpoints[i]) - np.array(endpoints[j]))
            if dist > max_dist:
                max_dist = dist
                best_pair = (endpoints[i], endpoints[j])

    # Weryfikacja czy punkty leżą na brzegu objętości
    if not is_on_surface(binary_volume, best_pair[0]) or not is_on_surface(binary_volume, best_pair[1]):
        return find_endpoints_by_distance(binary_volume)

    print("Punkty są na powierzchni")

    return best_pair


def is_on_surface(volume, point):
    """Sprawdza czy punkt leży na powierzchni objętości"""
    x, y, z = point
    for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
        nx, ny, nz = x + dx, y + dy, z + dz
        if (0 <= nx < volume.shape[0] and
                0 <= ny < volume.shape[1] and
                0 <= nz < volume.shape[2] and
                not volume[nx, ny, nz]):
            return True
    return False


def find_endpoints_by_distance(volume):
    """Alternatywna metoda znajdowania punktów końcowych oparta na maksymalnej odległości"""
    coords = np.argwhere(volume)

    # Znajdź punkt najbardziej wysunięty w osi Z (przyjęto, że aorta przebiega pionowo)
    z_sorted = coords[np.argsort(coords[:, 0])]
    top = tuple(z_sorted[0])
    bottom = tuple(z_sorted[-1])

    # Weryfikacja czy punkty są na powierzchni
    if not is_on_surface(volume, top) or not is_on_surface(volume, bottom):
        # Jeśli nie, znajdź punkty na powierzchni najbliżej ekstremów
        top = find_nearest_surface_point(volume, top)
        bottom = find_nearest_surface_point(volume, bottom)

    return top, bottom


def find_nearest_surface_point(volume, point):
    """Znajduje najbliższy punkt na powierzchni do danego punktu"""
    from scipy.spatial.distance import cdist
    surface_points = []
    coords = np.argwhere(volume)

    for voxel in coords:
        if is_on_surface(volume, tuple(voxel)):
            surface_points.append(voxel)

    if not surface_points:
        return point

    surface_points = np.array(surface_points)
    distances = cdist([point], surface_points)[0]
    nearest_idx = np.argmin(distances)
    return tuple(surface_points[nearest_idx])

def extract_path(pdef, start_voxel, end_voxel):
    path, _ = route_through_array(pdef, start_voxel, end_voxel, fully_connected=True)
    return np.array(path)

def smooth_centerline(path, sigma=2):
    # Poprawione wygładzanie dla każdej współrzędnej osobno
    smoothed = np.zeros_like(path, dtype=np.float32)
    for i in range(3):  # dla x, y, z
        smoothed[:, i] = gaussian_filter(path[:, i].astype(np.float32), sigma=sigma)
    return smoothed

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
    z, y, x = volume.shape

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
    input_vtp = r"D:\TOM_project_\0070_H_AO_H\0070_H_AO_H\Models\0183_1002_aorta.vtp"
    output_path = 'centerline.vtp'

    binary, origin, spacing = vtp_to_binary_volume(input_vtp, spacing=(0.01, 0.01, 0.01))
    show_slices(binary)
    cropped, offset = crop_to_colon(binary)

    # Upewnij się, że objętość jest dobrze wypełniona
    from scipy.ndimage import binary_fill_holes

    cropped = binary_fill_holes(cropped)

    start_voxel, end_voxel = estimate_endpoints(cropped)
    print(f"Znalezione punkty końcowe: Start {start_voxel}, End {end_voxel}")

    # Reszta procesu pozostaje bez zmian
    dbf = compute_dbf(cropped)
    gvf = compute_gvf(dbf)
    flags = flag_nonuniform_regions(gvf)
    connected = connect_flagged_voxels(flags)
    daf = compute_daf(connected)
    pdef = compute_pdef(daf, end_voxel)
    path = extract_path(pdef, start_voxel, end_voxel)
    smoothed = smooth_centerline(path)
    save_path_as_vtp(smoothed, origin=np.array(origin) + offset * spacing, spacing=spacing, output_path=output_path)
    visualize(cropped, origin=np.array(origin) + offset * spacing, spacing=spacing, path=smoothed)