import argparse
import os
import sys

sys.path.insert(0, ".")
import trimesh
from arap import Deformer
import json
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test data")
    parser.add_argument("name", type=str, help="Name of the test")
    parser.add_argument("mesh_path", type=str, help="Path to the mesh")

    args = parser.parse_args()

    os.makedirs("tests/" + args.name, exist_ok=True)

    mesh = trimesh.load(args.mesh_path)
    mesh.export(f"tests/{args.name}/input.off")

    d = Deformer()
    d.set_mesh(mesh.vertices, mesh.faces)
    np.save(f"tests/{args.name}/wm.npy", d.weight_matrix)
    deformation_matrix = np.load("tests/cube/t2.npy")
    np.save(f"tests/{args.name}/t2.npy", deformation_matrix)
    d.set_deformation(deformation_matrix)

    with open("tests/cube/sel.json", "r") as f:
        selection = json.load(f)
    d.set_selection(selection["selection"], selection["fixed"])
    with open(f"tests/{args.name}/sel.json", "w") as f:
        json.dump(selection, f)
    d.apply_deformation(10)

    mesh_o_gt = trimesh.load("tests/cube/output.off")
    verts = d.verts_prime
    faces = d.faces
    mesh_prime = trimesh.Trimesh(verts, faces)
    mesh_prime.export(f"tests/{args.name}/output.off")
