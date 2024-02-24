import unittest
from arap import Deformer
import json
import numpy as np
import trimesh
import torch


class TestArmadillo(unittest.TestCase):
    def test_armadillo(self):
        mesh = trimesh.load("tests/armadillo/input.off")

        d = Deformer()
        d.set_mesh(mesh.vertices, mesh.faces)
        weight_matrix_gt = np.load("tests/armadillo/wm.npy")
        weight_matrix = d.weight_matrix
        if type(weight_matrix) is torch.Tensor:
            weight_matrix = weight_matrix.cpu().numpy()
        weight_sum = d.weight_sum
        if type(weight_sum) is torch.Tensor:
            weight_sum = weight_sum.cpu().numpy()
        self.assertTrue(
            np.allclose(weight_matrix, weight_matrix_gt, atol=1e-3), "Weight matrix is not equal"
        )
        self.assertTrue(
            np.allclose(np.diagonal(weight_sum), np.sum(weight_matrix_gt, axis=1), atol=1e-3),
            "Weight sum is not equal",
        )
        deformation_matrix = np.load("tests/armadillo/t2.npy")
        d.set_deformation(deformation_matrix)

        with open("tests/armadillo/sel.json", "r") as f:
            selection = json.load(f)
        d.set_selection(selection["selection"], selection["fixed"])
        d.apply_deformation(10)

        verts_prime = d.verts_prime
        if type(verts_prime) is torch.Tensor:
            verts_prime = verts_prime.cpu().numpy()
        energy = d.calculate_energy()
        if type(energy) is torch.Tensor:
            energy = energy.cpu().numpy()

        mesh_o_gt = trimesh.load("tests/armadillo/output.off")
        self.assertTrue(
            np.allclose(mesh_o_gt.vertices, verts_prime, atol=1e-3), "Vertices are not equal"
        )
        self.assertTrue(np.isclose(energy, 6.0846, atol=1e-3), "Energy is not equal")


if __name__ == "__main__":
    unittest.main()
