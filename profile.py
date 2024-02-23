from arap import Deformer
import json
import numpy as np
import trimesh
import argparse
import cProfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARAP on a mesh")
    parser.add_argument("mesh_path", type=str, help="Path to the mesh file")

    args = parser.parse_args()

    mesh = trimesh.load(args.mesh_path)

    d = Deformer()
    d.set_mesh(mesh.vertices, mesh.faces)
    deformation_matrix = np.load("tests/t2.npy")
    d.set_deformation(deformation_matrix)

    with open("tests/sel.json", "r") as f:
        selection = json.load(f)
    d.set_selection(selection["selection"], selection["fixed"])

    cProfile.run("d.apply_deformation(100)", sort="cumtime")
