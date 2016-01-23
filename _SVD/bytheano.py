import theano 
from theano.tensor.nlinalg import svd 

a = theano.tensor.matrix('a')
y = svd(a)
SVD = theano.function([a], y, allow_input_downcast=True)

if __name__ == '__main__':
    import numpy as np
    N = 1000

    a = np.random.randint(0, 100, size=(N, N))
    u, s, v = SVD(a)
    print u, s, v


