import theano
import theano.tensor as T
from theano.tensor.nlinalg import diag as __gdiag
__gx = T.vector('__gx', dtype='float32') 
__gxs = T.matrix('__gxs', dtype='float32') 
__gM = T.matrix('__gM', dtype='float32') 
__gv = __gx-__gxs
__gdist = __gdiag(__gv.dot(__gM).dot(__gv.T))
__dist1vsN = theano.function([__gx, __gxs, __gM], __gdist, allow_input_downcast=True)

__gv = (__gx-__gxs)/T.sqrt(__gx+__gxs+1e-20)
__gdist = __gdiag(__gv.dot(__gM).dot(__gv.T))
__dist1vsNchisquare = theano.function([__gx, __gxs, __gM], __gdist, allow_input_downcast=True)

def __knn(train_x, test_x, train_y, test_y, M, K=5, verbose=False, cfmatrix=False):
    n = len(train_x)
    m = len(set(train_y))
    if M is None:
        M = np.eye(len(train_x[0]))
    acc = 0
    rec = np.zeros((m, m))
    for x, y in zip(test_x, test_y):
        dist = __dist1vsN(x, train_x, M) 

        predict = np.bincount(train_y[dist.argsort()[:K]]).argmax()
        if predict == y: acc += 1
        rec[y][predict] += 1
    if verbose:
        print '{}/{}'.format(acc, test_y.shape[0])

    acc = float(acc)/test_y.shape[0]*100 
    if cfmatrix:
        return acc, rec
    else:
        return acc


