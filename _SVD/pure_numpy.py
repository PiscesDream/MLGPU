from numpy.linalg import svd as SVD 

if __name__ == '__main__':
    import numpy as np
    N = 1000

    a = np.random.randint(0, 100, size=(N, N))
    u, s, v = npsvd(a)
    print u, s, v

