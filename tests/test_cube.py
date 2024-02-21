import unittest
from arap import Deformer
import json
import numpy as np
import trimesh


class TestCube(unittest.TestCase):
    def test_cube(self):
        d = Deformer("tests/cube3.obj")
        deformation_matrix = np.load("tests/t2.npy")
        d.set_deformation(deformation_matrix)

        with open("tests/sel.json", "r") as f:
            selection = json.load(f)
        d.set_selection(selection["selection"], selection["fixed"])
        d.calculate_laplacian_matrix()
        d.precompute_p_i()
        d.apply_deformation(100)
        d.output_s_prime_to_file()

        mesh_o_gt = trimesh.load("tests/output.off")
        self.assertTrue(
            np.allclose(mesh_o_gt.vertices, d.verts_prime, atol=1e-3), "Vertices are not equal"
        )


if __name__ == "__main__":
    unittest.main()