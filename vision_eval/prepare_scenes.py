"""Prepare STL chess pieces for scene rendering.

Loads and normalizes the eZeeOne Dubrovnik 1960 set 02 STL files, then scales
all pieces so the king is exactly KING_HEIGHT units tall.
"""

from pathlib import Path

import numpy as np
import vtk


ROOT = Path(__file__).resolve().parent
SQUARE = 50.0
KING_HEIGHT = 78.0
NAMES = ["pawn", "knight", "bishop", "rook", "queen", "king"]


def read_stl(path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.MergingOn()
    reader.Update()
    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(reader.GetOutput())
    if mesh.GetNumberOfPoints() == 0:
        raise ValueError("filemesh has no points: %s" % (path,))
    return mesh


def transform(mesh, matrix):
    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, float(matrix[i, j]))
    vtk_transform = vtk.vtkTransform()
    vtk_transform.SetMatrix(vtk_matrix)
    poly_filter = vtk.vtkTransformPolyDataFilter()
    poly_filter.SetInputData(mesh)
    poly_filter.SetTransform(vtk_transform)
    poly_filter.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(poly_filter.GetOutput())
    return out


def normalize(mesh):
    bounds = np.array(mesh.GetBounds(), dtype=float).reshape(3, 2)
    eye = np.eye(4, dtype=float)
    eye[0, 3] = -float(bounds[0].mean())
    eye[1, 3] = -float(bounds[1].mean())
    eye[2, 3] = -float(bounds[2, 0])
    return transform(mesh, eye)


def load_sets():
    prefix_map = {
        "pawn": "obj_5_",
        "knight": "obj_3_",
        "bishop": "obj_13_",
        "rook": "obj_4_",
        "queen": "obj_2_",
        "king": "obj_1_",
    }
    directory = ROOT / "prepared" / "ezeeone_Dubrovnik_1960_set_02_stls"
    pieces = {}
    for name in NAMES:
        prefix = prefix_map[name]
        matches = []
        for path in sorted(directory.glob("*.stl")):
            if path.name.startswith(prefix):
                matches.append(path)
        if len(matches) != 1:
            raise ValueError(
                "expected exactly 1 file for %s prefix %s, found %d"
                % (name, prefix, len(matches))
            )
        pieces[name] = normalize(read_stl(matches[0]))

    head_path = None
    head_matches = []
    for path in sorted(directory.glob("*.stl")):
        if path.name.startswith("obj_12_"):
            head_matches.append(path)
    if len(head_matches) != 1:
        raise ValueError(
            "expected exactly 1 obj_12_ file, found %d" % (len(head_matches),)
        )
    head_path = head_matches[0]

    head = normalize(read_stl(head_path))
    head_bounds = np.array(head.GetBounds(), dtype=float).reshape(3, 2)
    stem_height = float(np.ptp(head_bounds[2]) - np.ptp(head_bounds[0]))
    body = pieces["bishop"]
    body_out_bounds = np.array(body.GetBounds(), dtype=float).reshape(3, 2)
    matrix_z = float(body_out_bounds[2, 1] - stem_height)

    append = vtk.vtkAppendPolyData()
    append.AddInputData(body)
    shift = np.eye(4, dtype=float)
    shift[2, 3] = matrix_z
    append.AddInputData(transform(head, shift))
    append.Update()
    merged = vtk.vtkPolyData()
    merged.DeepCopy(append.GetOutput())
    pieces["bishop"] = merged

    king_bounds = np.array(pieces["king"].GetBounds(), dtype=float).reshape(3, 2)
    scale = float(KING_HEIGHT / king_bounds[2, 1])

    scaled = {}
    for name in NAMES:
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] *= scale
        scaled[name] = transform(pieces[name], matrix)

    return {"dubrovnik": scaled}, {"dubrovnik": scale}


def actor_for(mesh, color):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(color[0], color[1], color[2])
    prop.SetAmbient(0.22)
    prop.SetDiffuse(0.78)
    prop.SetSpecular(0.08)
    prop.SetSpecularPower(20.0)
    return actor


def plane(x0, y0, x1, y1, z=0):
    source = vtk.vtkPlaneSource()
    source.SetOrigin(x0, y0, z)
    source.SetPoint1(x1, y0, z)
    source.SetPoint2(x0, y1, z)
    source.Update()
    return source.GetOutput()
