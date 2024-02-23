import time
import trimesh
import numpy as np
import face_utils
import argparse
import othermath as omath
import matplotlib.pyplot as plt
import json
from tqdm.auto import tqdm
import torch
import dgl
from itertools import combinations

np.set_printoptions(precision=2, suppress=True)

SPARSE = False

if SPARSE:
    import scipy.sparse
    import scipy.sparse.linalg

    solve = scipy.sparse.linalg.spsolve
    matrix = scipy.sparse.lil_matrix
else:
    solve = np.linalg.solve
    matrix = np.zeros


# Read file into arrays
class Deformer:
    max_iterations = 100
    threshold = 0.001

    def __init__(self):
        self.POWER = float("Inf")
        self.stop_flag = False

    def get_edge_to_face_map(self, edges, faces):
        n = faces.max() + 1

        # Map from edge id to 2 face ids
        edge2face = np.zeros((n * n, 2), dtype=np.int32)
        # For each edge id, the number of faces it has been associated with so far
        edgec = np.zeros((n * n,), dtype=np.int8)

        for i in range(3):
            j = (i + 1) % 3
            e1, e2 = faces[:, i], faces[:, j]
            edge_id = e1 * n + e2
            edge2face[edge_id, edgec[edge_id]] = np.arange(len(faces))
            edgec[edge_id] += 1
            edge_id = e2 * n + e1
            edge2face[edge_id, edgec[edge_id]] = np.arange(len(faces))
            edgec[edge_id] += 1

        assert np.all(~((edgec > 0) & (edgec != 2))), "Some edges are not shared by exactly 2 faces"

        edge_id = edges[:, 0] * n + edges[:, 1]
        return edge2face[edge_id]

    def set_mesh(self, vertices, faces):
        self.n = len(vertices)
        self.vertices = vertices
        self.faces = faces
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]), axis=0)
        edges = np.concatenate((edges, np.flip(edges, axis=1)), axis=0)
        self.edges = np.unique(edges, axis=0)

        self.graph = dgl.graph(self.edges.tolist())
        self.graph.ndata["id"] = torch.arange(len(self.vertices))
        self.graph.ndata["verts"] = torch.from_numpy(vertices).to(torch.float32)
        self.graph.ndata["verts_prime"] = torch.from_numpy(vertices).to(torch.float32)
        self.graph.ndata["cell_rotations"] = torch.zeros((self.n, 3, 3), dtype=torch.float32)
        self.graph.edata["f"] = torch.from_numpy(
            self.get_edge_to_face_map(self.edges, self.faces)
        ).to(torch.int32)

        def weight_matrix(edges):
            src_id, dst_id = edges.src["id"].numpy(), edges.dst["id"].numpy()
            f = edges.data["f"].numpy()
            data = self.weight_for_pair(src_id, dst_id, f)

            return {"w": data}

        print("Precomputing weights...")
        self.graph.apply_edges(weight_matrix)
        print("Precomputing weights done")

        def weight_sum(node):
            return {"w_sum": torch.sum(node.mailbox["w"], 1)}

        self.graph.update_all(dgl.function.copy_e("w", "w"), weight_sum)
        self.weight_matrix = torch.zeros((self.n, self.n), dtype=torch.float32)
        edges = self.graph.adj().indices()
        self.weight_matrix[edges[0], edges[1]] = self.graph.edata["w"]

        self.weight_sum = torch.diag(self.graph.ndata["w_sum"])

    @property
    def cell_rotations(self):
        return self.graph.ndata["cell_rotations"]

    def faces_for_edge(self, i, j):
        return [f for f in self.faces if i in f and j in f]

    def other_point(self, face, i, j):
        """
        face: (n, 3)
        i: (n)
        j: (n)
        """
        mask = (face != i[:, None]) & (face != j[:, None])
        other_point = face[mask]
        assert len(other_point) == len(i)
        return other_point

    def weight_for_pair(self, i, j, f):
        """
        i: (n,)
        j: (n,)
        f: (n, 2)
        """

        vertex_i = self.graph.ndata["verts"][i]  # (n, 3)
        vertex_j = self.graph.ndata["verts"][j]  # (n, 3)
        faces = self.faces[f]  # (n, 2, 3)

        w = torch.zeros((len(i),), dtype=torch.float32)
        for fn in range(2):
            face = faces[:, fn]
            other_vertex_id = self.other_point(face, i, j)
            vertex_o = self.graph.ndata["verts"][other_vertex_id]  # (n, 3)
            e1 = vertex_i - vertex_o
            e2 = vertex_j - vertex_o
            theta = np.arccos(
                (e1 * e2).sum(1) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1))
            )
            w += np.cos(theta) / np.sin(theta) * 0.5

        return w

    def assign_values_to_neighbour_matrix(self, v1, v2, v3):
        self.neighbour_matrix[v1, v2] = 1
        self.neighbour_matrix[v2, v1] = 1
        self.neighbour_matrix[v1, v3] = 1
        self.neighbour_matrix[v3, v1] = 1
        self.neighbour_matrix[v2, v3] = 1
        self.neighbour_matrix[v3, v2] = 1

    def reset(self):
        self.graph.ndata["verts_prime"] = self.graph.ndata["verts"].clone()

    def set_selection(self, selection_ids, fixed_ids):
        self.vert_status = [1] * self.n
        self.fixed_verts = []
        self.selected_verts = []
        for i in selection_ids:
            self.vert_status[i] = 2
            self.selected_verts.append(i)
            self.fixed_verts.append(
                (i, omath.apply_rotation(self.deformation_matrix, self.graph.ndata["verts"][i]))
            )
        for i in fixed_ids:
            self.vert_status[i] = 0
            self.fixed_verts.append((i, self.graph.ndata["verts"][i]))

    # Reads the .sel file and keeps track of the selection status of a vertex
    def read_selection_file(self, filename):
        # The selection status of each vertex, where 0=fixed, 1=deformable-region, 2=handle

        self.vert_status = open(filename, "r").read().strip().split("\n")
        # Remove any lines that aren't numbers
        self.vert_status = [int(line) for line in self.vert_status if omath.string_is_int(line)]

        # Keep track of the IDs of the selected verts (i.e. verts with handles/status == 2)
        self.selected_verts = []
        self.fixed_verts = []
        for i in range(self.n):
            if self.vert_status[i] == 2:
                self.selected_verts.append(i)
                self.fixed_verts.append(
                    (i, omath.apply_rotation(self.deformation_matrix, self.graph.ndata["verts"][i]))
                )
            elif self.vert_status[i] == 0:
                self.fixed_verts.append((i, self.graph.ndata["verts"][i]))
        assert len(self.vert_status) == len(self.graph.ndata["verts"])

    def set_deformation(self, deformation_matrix):
        self.deformation_matrix = deformation_matrix

    # Reads the .def file and stores the inner matrix
    def read_deformation_file(self, filename):
        def_file_lines = open(filename, "r").read().strip().split("\n")
        # Remove any lines with comments
        def_file_lines = [line for line in def_file_lines if "#" not in line]

        # Assert its at least a 4 by something matrix just in case
        assert len(def_file_lines) == 4
        self.deformation_matrix = np.matrix(";".join(def_file_lines))
        print("Deformation matrix to apply")
        print(self.deformation_matrix)
        assert self.deformation_matrix.size == 16

    # Returns a set of IDs that are neighbours to this vertexID (not including the input ID)
    def neighbours_of(self, vert_id):
        return self.graph.out_edges(vert_id)[1].numpy()

    def calculate_laplacian_matrix(self):
        # initial laplacian
        # self.laplacian_matrix = self.edge_matrix - self.neighbour_matrix
        self.laplacian_matrix = self.weight_sum - self.weight_matrix
        fixed_verts_num = len(self.fixed_verts)
        # for each constrained point, add a new row and col
        new_n = self.n + fixed_verts_num
        new_matrix = matrix((new_n, new_n), dtype=np.float32)
        # Assign old values to new matrix
        new_matrix[: self.n, : self.n] = self.laplacian_matrix
        # Add 1s in the row and column associated with the fixed point to constain it
        # This will increase L by the size of fixed_verts
        for i in range(fixed_verts_num):
            new_i = self.n + i
            vert_id = self.fixed_verts[i][0]
            new_matrix[new_i, vert_id] = 1
            new_matrix[vert_id, new_i] = 1

        self.laplacian_matrix = new_matrix

    def stop(self):
        self.stop_flag = True

    def apply_deformation(self, iterations):
        print("Precomputing laplacian matrix...")
        self.calculate_laplacian_matrix()
        print("Precomputing laplacian done")
        print("Length of sel verts", len(self.selected_verts))

        if iterations < 0:
            iterations = self.max_iterations

        self.current_energy = 0

        # initialize b and assign constraints
        number_of_fixed_verts = len(self.fixed_verts)

        self.b_array = np.zeros((self.n + number_of_fixed_verts, 3))
        # Constraint b points
        for i in range(number_of_fixed_verts):
            self.b_array[self.n + i] = self.fixed_verts[i][1]

        # Apply following deformation iterations
        pbar = tqdm(range(iterations))
        for t in pbar:
            if self.stop_flag:
                self.stop_flag = False
                break
            # print("Iteration: ", t)

            self.calculate_cell_rotations()
            self.apply_cell_rotations()
            iteration_energy = self.calculate_energy()
            pbar.set_description(f"Energy: {self.current_energy:.2f}")
            # print("Total Energy: ", self.current_energy)
            # if(self.energy_minimized(iteration_energy)):
            #     print("Energy was minimized at iteration", t, " with an energy of ", iteration_energy)
            #     break
            self.current_energy = iteration_energy

    def energy_minimized(self, iteration_energy):
        return abs(self.current_energy - iteration_energy) < self.threshold

    def calculate_cell_rotations(self):
        def rot_message(edges):
            return {
                "verts": edges.dst["verts"],
                "verts_prime": edges.dst["verts_prime"],
                "w": edges.data["w"],
                "verts_in": edges.src["verts"],
                "verts_prime_in": edges.src["verts_prime"],
            }

        def cacl_cell_rotation(nodes):
            rotations = self.calculate_rotation_matrix_for_cell(
                nodes.mailbox["verts"][:, 0],
                nodes.mailbox["verts_prime"][:, 0],
                nodes.mailbox["w"],
                nodes.mailbox["verts_in"],
                nodes.mailbox["verts_prime_in"],
            )
            return {"cell_rotations": rotations}

        self.graph.update_all(rot_message, cacl_cell_rotation)

    def vert_is_deformable(self, vert_id):
        return self.vert_status[vert_id] == 1

    def calculate_rotation_matrix_for_cell(
        self, verts, verts_prime, w_in, verts_in, verts_prime_in
    ):
        """
        verts: (n, 3)
        verts_prime: (n, 3)
        w_in: (n, d_i)
        verts_in: (n, d_i, 3)
        verts_prime_in: (n, d_i, 3)
        """
        # covariance_matrix = self.calculate_covariance_matrix_for_cell(vert_id)
        covariance_matrix = self.calculate_batched_covariance_matrix_for_cell(
            verts, verts_prime, w_in, verts_in, verts_prime_in
        )  # (n, 3, 3)

        U, s, V_transpose = torch.linalg.svd(covariance_matrix)  # (n, 3, 3), (n, 3), (n, 3, 3)

        det_sign = torch.linalg.det(V_transpose @ U)  # (n,)
        U[:, 0] *= det_sign[:, None]
        rotation = V_transpose.swapaxes(1, 2) @ U.swapaxes(1, 2)  # (n, 3, 3)
        return rotation

    def calculate_batched_covariance_matrix_for_cell(
        self, verts, verts_prime, w_in, verts_in, verts_prime_in
    ):
        """
        verts: (n, 3)
        verts_prime: (n, 3)
        w_in: (n, d_i)
        verts_in: (n, d_i, 3)
        verts_prime_in: (n, d_i, 3)
        """
        D_i = torch.diag_embed(w_in)  # (n, d_i, d_i)

        P_i = verts[:, None] - verts_in  # (n, d_i, 3)
        P_i_prime = verts_prime[:, None] - verts_prime_in  # (n, d_i, 3)

        return torch.einsum("nai,nab,nbj->nij", P_i, D_i, P_i_prime)  # (n, 3, 3)

    def output_s_prime_to_file(self):
        # Write self.vers_prime and self.faces to a file
        mesh = trimesh.Trimesh(self.graph.ndata["verts_prime"], self.faces)
        mesh.export("output.off")

    def apply_cell_rotations(self):
        # print("Applying Cell Rotations")
        def b_message(edges):
            return {
                "verts": edges.dst["verts"],
                "verts_in": edges.src["verts"],
                "R": edges.dst["cell_rotations"],
                "R_in": edges.src["cell_rotations"],
                "w_in": edges.data["w"],
            }

        def calc_b(nodes):
            b = self.calculate_b_for(
                nodes.mailbox["verts"] - nodes.mailbox["verts_in"],
                nodes.mailbox["R"][:, 0],
                nodes.mailbox["R_in"],
                nodes.mailbox["w_in"],
            )
            return {"b": b}

        self.graph.update_all(b_message, calc_b)
        self.b_array[: self.n] = self.graph.ndata["b"].numpy()

        self.graph.ndata["verts_prime"] = torch.from_numpy(
            solve(self.laplacian_matrix, self.b_array)[: self.n]
        ).to(torch.float32)

    def calculate_b_for(self, P, R, R_in, w_in):
        """
        P: (n, d_i, 3)
        R: (n, 3, 3)
        R_in: (n, d_i, 3, 3)
        w_in: (n, d_i)
        """
        R_avg = R[:, None] + R_in  # (n, d_i, 3, 3)
        b = (torch.einsum("nijk,nik->nij", R_avg, P) * w_in[..., None]).sum(axis=1) / 2

        return b

    def calculate_energy(self):
        total_energy = 0
        for i in range(self.n):
            total_energy += self.energy_of_cell(i)
        return total_energy

    def energy_of_cell(self, i):
        neighbours = self.neighbours_of(i)
        total_energy = 0
        for j in neighbours:
            w_ij = self.weight_matrix[i, j]
            e_ij_prime = self.graph.ndata["verts_prime"][i] - self.graph.ndata["verts_prime"][j]
            e_ij = self.graph.ndata["verts"][i] - self.graph.ndata["verts"][j]
            r_i = self.cell_rotations[i]
            value = e_ij_prime - r_i @ e_ij
            if self.POWER == float("Inf"):
                norm_power = torch.max(torch.abs(value))
            else:
                norm_power = np.power(value, self.POWER)
                norm_power = np.sum(norm_power)
            # total_energy += w_ij * np.linalg.norm(, ord=self.POWER) ** self.POWER
            total_energy += w_ij * norm_power
        return total_energy

    @property
    def verts_prime(self):
        return self.graph.ndata["verts_prime"]

    def hex_color_for_energy(self, energy, max_energy):
        relative_energy = (energy / max_energy) * 255
        relative_energy = max(0, min(int(relative_energy), 255))
        red = hex(relative_energy)[2:]
        blue = hex(255 - relative_energy)[2:]
        if len(red) == 1:
            red = "0" + red
        if len(blue) == 1:
            blue = "0" + blue
        return "#" + red + "00" + blue

    def hex_color_array(self):
        energies = [self.energy_of_cell(i) for i in range(self.n)]
        max_value = np.amax(energies)
        return [self.hex_color_for_energy(energy, max_value) for energy in energies]

    def show_graph(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        xs = np.squeeze(np.asarray(self.graph.ndata["verts_prime"][:, 0]))
        ys = np.squeeze(np.asarray(self.graph.ndata["verts_prime"][:, 1]))
        zs = np.squeeze(np.asarray(self.graph.ndata["verts_prime"][:, 2]))
        color = self.hex_color_array()
        # Axes3D.scatter(xs, ys, zs=zs, zdir='z', s=1)#, c=None, depthshade=True, *args, **kwargs)
        ax.scatter(xs, ys, zs, c=color)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deform a mesh using ARAP")

    parser.add_argument("--filename", "-f", type=str, help="The .off file to deform", required=True)
    parser.add_argument("--selection", "-s", type=str, help="The .sel file to use", required=True)
    parser.add_argument(
        "--deformation",
        "-d",
        type=str,
        help="The .npy file with the 4x4 deformation matrix",
        required=True,
    )
    parser.add_argument(
        "--iterations", "-i", type=int, help="The number of iterations to run", default=-1
    )

    args = parser.parse_args()

    filename = args.filename
    selection_filename = args.selection
    deformation_file = args.deformation
    iterations = args.iterations

    t = time.time()

    d = Deformer()

    mesh = trimesh.load_mesh(filename)

    d.set_mesh(mesh.vertices, mesh.faces)

    deformation_matrix = np.load(deformation_file)
    d.set_deformation(deformation_matrix)

    with open(selection_filename, "r") as f:
        selection = json.load(f)
    d.set_selection(selection["selection"], selection["fixed"])

    print("Precomputation time ", time.time() - t)
    t = time.time()
    d.apply_deformation(iterations)
    print("Total iteration time", time.time() - t)
    d.output_s_prime_to_file()
    d.show_graph()

    print(d.vert_status)
