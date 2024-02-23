import numpy as np
import math
import torch


def angle_between(vector_a, vector_b):
    costheta = vector_a.dot(vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    return math.acos(costheta)


def cot(theta):
    return math.cos(theta) / math.sin(theta)


def string_is_int(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def to_homogeneous(v):
    if type(v) is np.ndarray:
        v = np.concatenate((v, np.ones((*v.shape[:-1], 1))), axis=-1)
    elif type(v) is torch.Tensor:
        v = torch.cat((v, torch.ones((*v.shape[:-1], 1)).to(v.device)), dim=-1)
    return v


def inf_norm(matrix):
    return np.amax(np.abs(matrix))


# Apply the 4x4 matrix to the 1x3 vector
