import numpy as np


def computeHomography(points1, points2):
    assert(len(points1) == 4)
    assert(len(points2) == 4)
    n = 8
    m = 9
    A = np.zeros((n, m))
    for i in range(int(n/2)):
        p = points1[i]
        q = points2[i]
        A[i * 2][0] = -p[0]
        A[i * 2 + 1][3] = -p[0]
        A[i * 2][1] = -p[1]
        A[i * 2 + 1][4] = -p[1]
        A[i * 2][2] = -1
        A[i * 2 + 1][5] = -1
        A[i * 2][6] = p[0] * q[0]
        A[i * 2][7] = p[1] * q[0]
        A[i * 2][8] = q[0]
        A[i * 2 + 1][6] = p[0] * q[1]
        A[i * 2 + 1][7] = p[1] * q[1]
        A[i * 2 + 1][8] = q[1]
    u, s, v = np.linalg.svd(A, full_matrices=True)
    H = v[8].reshape(-1, 3) / v[-1][-1]
    return H
