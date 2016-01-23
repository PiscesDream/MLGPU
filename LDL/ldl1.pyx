import numpy as np

def LDL(A):
    assert(A.shape[0] == A.shape[1])

    L = np.zeros(A.shape)
    D = np.zeros(A.shape)
    n = A.shape[0]

    cdef int i 
    cdef int j
    cdef int k

    for i from 0 <= i < n:
        for j from 0 <= j < n:
            if i == j:
                D[i, i] = A[i, i]
                for k from 0 <= k < i:
                    D[i, i] -= (L[i, k] ** 2) * D[k, k]
                L[i, i] = 1
            elif j <= i:
                L[i, j] = A[i, j]
                for k from 0 <= k < j:
                    L[i, j] -= L[i, k] * D[k, k] * L[j, k]
                L[i, j] *= 1/D[j, j]

    return L, D
