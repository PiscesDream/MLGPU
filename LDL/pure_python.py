import numpy as np

def LDL(A):
    assert(A.shape[0] == A.shape[1])

    L = np.zeros(A.shape)
    D = np.zeros(A.shape)
    n = A.shape[0]

    for i in xrange(n):
        for j in xrange(n):
            if i == j:
                D[i, i] = A[i, i]
                for k in xrange(i):
                    D[i, i] -= (L[i, k] ** 2) * D[k, k]
                L[i, i] = 1
            elif j <= i:
                L[i, j] = A[i, j]
                for k in xrange(j):
                    L[i, j] -= L[i, k] * D[k, k] * L[j, k]
                L[i, j] *= 1/D[j, j]

    return L, D

if __name__ == '__main__':
    a = np.array([[1,2,3], [3,2,4]])
    M = np.dot(a.T, a)
    L, D = LDL(M)

    H = np.dot(L, np.sqrt(D[:,:linalg.matrix_rank(M)]))


