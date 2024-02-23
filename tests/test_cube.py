import unittest
from arap import Deformer
import json
import numpy as np
import trimesh


class TestCube(unittest.TestCase):
    def test_cube(self):
        mesh = trimesh.load("tests/cube3.obj")

        d = Deformer()
        d.set_mesh(mesh.vertices, mesh.faces)
        weight_matrix_gt = np.load("tests/cube_wm.npy")
        self.assertTrue(
            np.allclose(d.weight_matrix, weight_matrix_gt, atol=1e-3), "Weight matrix is not equal"
        )
        self.assertTrue(
            np.allclose(np.diagonal(d.weight_sum), np.sum(weight_matrix_gt, axis=1), atol=1e-3),
            "Weight sum is not equal",
        )
        deformation_matrix = np.load("tests/t2.npy")
        d.set_deformation(deformation_matrix)

        with open("tests/sel.json", "r") as f:
            selection = json.load(f)
        d.set_selection(selection["selection"], selection["fixed"])
        d.apply_deformation(100)
        d.output_s_prime_to_file()

        mesh_o_gt = trimesh.load("tests/output.off")
        self.assertTrue(
            np.allclose(mesh_o_gt.vertices, d.verts_prime, atol=1e-3), "Vertices are not equal"
        )
        self.assertTrue(np.isclose(d.calculate_energy(), 6.0846, atol=1e-3), "Energy is not equal")
        print()


if __name__ == "__main__":
    unittest.main()
