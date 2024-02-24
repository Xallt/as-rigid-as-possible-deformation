import unittest
from arap import Deformer
import json
import numpy as np
import trimesh


class TestArmadillo(unittest.TestCase):
    def test_armadillo(self):
        mesh = trimesh.load("tests/armadillo/input.off")

        d = Deformer()
        d.set_mesh(mesh.vertices, mesh.faces)
        weight_matrix_gt = np.load("tests/armadillo/wm.npy")
        self.assertTrue(
            np.allclose(d.weight_matrix, weight_matrix_gt, atol=1e-3), "Weight matrix is not equal"
        )
        self.assertTrue(
            np.allclose(np.diagonal(d.weight_sum), np.sum(weight_matrix_gt, axis=1), atol=1e-3),
            "Weight sum is not equal",
        )
        deformation_matrix = np.load("tests/armadillo/t2.npy")
        d.set_deformation(deformation_matrix)

        with open("tests/armadillo/sel.json", "r") as f:
            selection = json.load(f)
        d.set_selection(selection["selection"], selection["fixed"])
        d.calculate_laplacian_matrix()
        d.precompute_p_i()
        d.apply_deformation(10)

        mesh_o_gt = trimesh.load("tests/armadillo/output.off")
        self.assertTrue(
            np.allclose(mesh_o_gt.vertices, d.verts_prime, atol=1e-3), "Vertices are not equal"
        )


if __name__ == "__main__":
    unittest.main()
