import time
import trimesh
import numpy as np
import face_utils
import argparse
import othermath as omath
import matplotlib.pyplot as plt
import json
from tqdm.auto import tqdm

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

    def __init__(self, filename):
        self.filename = filename
        self.POWER = float("Inf")

        self.read_file()
        self.build_weight_matrix()

    def read_file(self):
        # Load the mesh file with trimesh
        mesh = trimesh.load(self.filename, force="mesh")

        # Initialize object fields
        self.n = len(mesh.vertices)
        self.verts = mesh.vertices
        self.verts_prime = np.array(mesh.vertices)  # Copy vertices as verts_prime
        self.faces = mesh.faces

        # Initialize verts_to_face
        self.verts_to_face = [[] for _ in range(self.n)]
        for i, face in enumerate(self.faces):
            for vert_id in face:
                self.verts_to_face[vert_id].append(i)

        # Neighbour matrix and edge matrix calculations
        self.neighbour_matrix = np.zeros((self.n, self.n))
        for face in self.faces:
            for i in range(3):
                for j in range(i + 1, 3):
                    self.neighbour_matrix[face[i], face[j]] = 1
                    self.neighbour_matrix[face[j], face[i]] = 1

        print("Generating Edge Matrix")
        self.edge_matrix = np.zeros((self.n, self.n))
        for row in range(self.n):
            self.edge_matrix[row][row] = np.sum(self.neighbour_matrix[row])

        print("Generating Laplacian Matrix")
        # Assuming you're calculating the Laplacian matrix; replace with your method
        self.laplacian_matrix = (
            np.diag(np.sum(self.neighbour_matrix, axis=1)) - self.neighbour_matrix
        )

        # N size array of 3x3 matrices for cell rotations
        self.cell_rotations = np.zeros((self.n, 3, 3))

        print(f"{len(self.verts)} vertices")
        print(f"{len(self.faces)} faces")
        print(f"{np.sum(self.neighbour_matrix) // 2} edges")  # Assuming undirected edges

    def assign_values_to_neighbour_matrix(self, v1, v2, v3):
        self.neighbour_matrix[v1, v2] = 1
        self.neighbour_matrix[v2, v1] = 1
        self.neighbour_matrix[v1, v3] = 1
        self.neighbour_matrix[v3, v1] = 1
        self.neighbour_matrix[v2, v3] = 1
        self.neighbour_matrix[v3, v2] = 1

    def set_selection(self, selection_ids, fixed_ids):
        self.vert_status = [1] * self.n
        self.fixed_verts = []
        self.selected_verts = []
        for i in selection_ids:
            self.vert_status[i] = 2
            self.selected_verts.append(i)
            self.fixed_verts.append(
                (i, omath.apply_rotation(self.deformation_matrix, self.verts[i]))
            )
        for i in fixed_ids:
            self.vert_status[i] = 0
            self.fixed_verts.append((i, self.verts[i]))

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
                    (i, omath.apply_rotation(self.deformation_matrix, self.verts[i]))
                )
            elif self.vert_status[i] == 0:
                self.fixed_verts.append((i, self.verts[i]))
        assert len(self.vert_status) == len(self.verts)

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
        return np.where(self.neighbour_matrix[vert_id])[0]

    def build_weight_matrix(self):
        print("Generating Weight Matrix")
        self.weight_matrix = matrix((self.n, self.n), dtype=np.float32)
        self.weight_sum = matrix((self.n, self.n), dtype=np.float32)

        for vertex_id in range(self.n):
            neighbours = self.neighbours_of(vertex_id)
            for neighbour_id in neighbours:
                self.assign_weight_for_pair(vertex_id, neighbour_id)
        print(self.weight_matrix)

    def assign_weight_for_pair(self, i, j):
        if self.weight_matrix[j, i] == 0:
            # If the opposite weight has not been computed, do so
            weightIJ = self.weight_for_pair(i, j)
        else:
            weightIJ = self.weight_matrix[j, i]
        self.weight_sum[i, i] += weightIJ * 0.5
        self.weight_sum[j, j] += weightIJ * 0.5
        self.weight_matrix[i, j] = weightIJ

    def weight_for_pair(self, i, j):
        local_faces = []
        # For every face associated with vert index I,
        for f_id in self.verts_to_face[i]:
            face = self.faces[f_id]
            # If the face contains both I and J, add it
            if i in face and j in face:
                local_faces.append(face)

        # Either a normal face or a boundry edge, otherwise bad mesh
        assert len(local_faces) <= 2

        vertex_i = self.verts[i]
        vertex_j = self.verts[j]

        # weight equation: 0.5 * (cot(alpha) + cot(beta))

        cot_theta_sum = 0
        for face in local_faces:
            other_vertex_id = face_utils.other_point(face, i, j)
            vertex_o = self.verts[other_vertex_id]
            theta = omath.angle_between(vertex_i - vertex_o, vertex_j - vertex_o)
            cot_theta_sum += omath.cot(theta)
        return cot_theta_sum * 0.5

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
        print(self.laplacian_matrix)

        self.laplacian_matrix = new_matrix

    def apply_deformation(self, iterations):
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
        # print("Calculating Cell Rotations")
        for vert_id in range(self.n):
            rotation = self.calculate_rotation_matrix_for_cell(vert_id)
            self.cell_rotations[vert_id] = rotation

    def vert_is_deformable(self, vert_id):
        return self.vert_status[vert_id] == 1

    def precompute_p_i(self):
        self.P_i_array = []
        for i in range(self.n):
            vert_i = self.verts[i]
            neighbour_ids = self.neighbours_of(i)
            number_of_neighbours = len(neighbour_ids)

            P_i = np.zeros((3, number_of_neighbours))

            for n_i in range(number_of_neighbours):
                n_id = neighbour_ids[n_i]

                vert_j = self.verts[n_id]
                P_i[:, n_i] = vert_i - vert_j
            self.P_i_array.append(P_i)

    def apply_cell_rotations(self):
        # print("Applying Cell Rotations")

        # Regular b points
        for i in range(self.n):
            self.b_array[i] = self.calculate_b_for(i)

        # print("Printing B")
        # print(self.b_array)

        p_prime = solve(self.laplacian_matrix, self.b_array)

        # self.verts = self.verts_prime

        for i in range(self.n):
            self.verts_prime[i] = p_prime[i]

        # print("p prime")
        # print(p_prime)

    def calculate_rotation_matrix_for_cell(self, vert_id):
        covariance_matrix = self.calculate_covariance_matrix_for_cell(vert_id)

        U, s, V_transpose = np.linalg.svd(covariance_matrix)

        # U, s, V_transpose
        # V_transpose_transpose * U_transpose

        rotation = V_transpose.transpose().dot(U.transpose())
        if np.linalg.det(rotation) <= 0:
            U[:0] *= -1
            rotation = V_transpose.transpose().dot(U.transpose())
        return rotation

    def calculate_covariance_matrix_for_cell(self, vert_id):
        # s_i = P_i * D_i * P_i_prime_transpose
        vert_i_prime = self.verts_prime[vert_id]  # (N, 3)

        neighbour_ids = self.neighbours_of(vert_id)  # (d_i)

        D_i = np.diag(self.weight_matrix[vert_id, neighbour_ids])  # (d_i, d_i)

        P_i = self.P_i_array[vert_id]  # (3, d_i)
        P_i_prime = vert_i_prime[:, None] - self.verts_prime[neighbour_ids].T  # (3, d_i)

        P_i_prime = P_i_prime.transpose()
        return P_i @ D_i @ P_i_prime  # (3, 3)

    def output_s_prime_to_file(self):
        # Write self.vers_prime and self.faces to a file
        mesh = trimesh.Trimesh(self.verts_prime, self.faces)
        mesh.export("output.off")

    def calculate_b_for(self, i):
        b = np.zeros((1, 3))
        neighbours = self.neighbours_of(i)
        R = self.cell_rotations[neighbours]  # (d_i, 3, 3)
        R_avg = self.cell_rotations[i][None] + R  # (d_i, 3, 3)
        P = self.P_i_array[i].T  # (d_i, 3)
        b = (np.einsum("ijk,ik->ij", R_avg, P) * self.weight_matrix[i, neighbours][:, None]).sum(
            axis=0
        ) / 2

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
            e_ij_prime = self.verts_prime[i] - self.verts_prime[j]
            e_ij = self.verts[i] - self.verts[j]
            r_i = self.cell_rotations[i]
            value = e_ij_prime - r_i.dot(e_ij)
            if self.POWER == float("Inf"):
                norm_power = omath.inf_norm(value)
            else:
                norm_power = np.power(value, self.POWER)
                norm_power = np.sum(norm_power)
            # total_energy += w_ij * np.linalg.norm(, ord=self.POWER) ** self.POWER
            total_energy += w_ij * norm_power
        return total_energy

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
        xs = np.squeeze(np.asarray(self.verts_prime[:, 0]))
        ys = np.squeeze(np.asarray(self.verts_prime[:, 1]))
        zs = np.squeeze(np.asarray(self.verts_prime[:, 2]))
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

    d = Deformer(filename)

    deformation_matrix = np.load(deformation_file)
    d.set_deformation(deformation_matrix)

    with open(selection_filename, "r") as f:
        selection = json.load(f)
    d.set_selection(selection["selection"], selection["fixed"])

    d.calculate_laplacian_matrix()
    d.precompute_p_i()
    print("Precomputation time ", time.time() - t)
    t = time.time()
    d.apply_deformation(iterations)
    print("Total iteration time", time.time() - t)
    d.output_s_prime_to_file()
    d.show_graph()
