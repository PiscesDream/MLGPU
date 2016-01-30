import theano
import theano.tensor as T
import theano.tensor.nlinalg as linalg

def SPD_Project(mat):
    # force symmetric
    mat = (mat+mat.T)/2.0
    eig, eigv = linalg.eig(mat)
    eig = T.maximum(eig, 0)
    eig = T.diag(eig)
    return eigv.dot(eig).dot(eigv.T)


