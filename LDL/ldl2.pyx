import numpy as np
import numpy.linalg as linalg
cimport numpy as np
dtype = np.float64
ctypedef np.float64_t dtype_t

cimport cython
@cython.boundscheck(False)
def fastLDL(np.ndarray[dtype_t, ndim=2] A):
    assert(A.shape[0] == A.shape[1])

    cdef unsigned int i, j, k
    cdef int n = A.shape[0]
    cdef np.ndarray[dtype_t, ndim=2] L = np.zeros((n, n))
    cdef np.ndarray[dtype_t, ndim=2] D = np.zeros((n, n))

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

def LDL(A, combined=False):
    L, D = fastLDL( np.asarray(A, dtype=dtype) )

    if combined:
        return np.dot(L, np.sqrt(D[:,:linalg.matrix_rank(A)]))
    else:
        return L, D

