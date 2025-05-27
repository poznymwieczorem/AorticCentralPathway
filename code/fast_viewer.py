import vtk

file_path = r"D:\TOM_project_\0143_H_AO_H\Models\KDR34_aorta.vtp"

reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(file_path)

try:
    reader.Update()
    polydata = reader.GetOutput()
    print("Liczba punktów:", polydata.GetNumberOfPoints())
    print("Liczba komórek:", polydata.GetNumberOfCells())
except Exception as e:
    print("❌ Nie udało się wczytać pliku:", str(e))