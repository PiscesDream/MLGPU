import numpy as np
import cPickle

import theano
import theano.tensor as T
import theano.tensor.nlinalg as linalg

# if recursive depth is reached
#import sys
#sys.setrecursionlimit(10000)

# I wish scipy or numpy can have LDL in their linalg
# You can accelerate it with Cython or Fortran
def __LDL__(A, combined=False):
    import numpy.linalg as nplinalg
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

    if combined:
        return np.dot(L, np.sqrt(D[:,:nplinalg.matrix_rank(A)]))
    else:
        return L, D


class LMNN(object):
    def __init__(self, k=3, mu=0.5, verbose=False, autosave=True): 
        self.k = k
        self.mu = mu
        self.built = False
        self.trained = False
        self.verbose = verbose
        self.autosave = autosave

        self._x = T.matrix('_x', dtype='float32')
        self._y = T.ivector('_y') 
        self._lr = T.scalar('_lr', dtype='float32')
        self._set = T.imatrix('_set')
        self._neighborpairs = T.imatrix('_neighborpairs')

    def build(self, dim):
        M = theano.shared(value=np.eye(dim, dtype='float32'), name='M', borrow=True)

        pull_error = 0.
        ivectors = self._x[self._neighborpairs[:, 0]]
        jvectors = self._x[self._neighborpairs[:, 1]]
        diffv = ivectors - jvectors
        pull_error = linalg.trace(diffv.dot(M).dot(diffv.T))

        push_error = 0.0
        ivectors = self._x[self._set[:, 0]]
        jvectors = self._x[self._set[:, 1]]
        lvectors = self._x[self._set[:, 2]]
        diffij = ivectors - jvectors
        diffil = ivectors - lvectors
        lossij = diffij.dot(M).dot(diffij.T)
        lossil = diffil.dot(M).dot(diffil.T)
        mask = T.neq(self._y[self._set[:, 0]], self._y[self._set[:, 2]])
        push_error = linalg.trace(mask*T.maximum(lossij - lossil + 1, 0))

        error = (1-self.mu) * pull_error + self.mu * push_error
        updates = [(M, M - self._lr * T.grad(error, M))]

        self.M = M
        self.updates = updates
        self.pull_error = pull_error
        self.push_error = push_error 
        self.built = True

    def fit(self, x, y, lr=1e-7, maxS=1000, max_iter=100, reset=20): 
        self.maxS = maxS
        if self.built:
            M = self.M
        else:
            self.build(x.shape[1])
            M = self.M

        self.train_model = theano.function(
            [self._set, self._neighborpairs, self._lr], 
            T.stack([self.pull_error, self.push_error]),
            updates = self.updates,
            givens = {self._x: np.asarray(x, dtype='float32'),
                      self._y: np.asarray(y, dtype='int32')},

            allow_input_downcast=True
            )

        self.theano_SPDproject = theano.function([], [], 
            updates=[(M, self.__theano_SPDproject(M))])


        #_lr = lr
        t = 0
        neighbors = self._get_neighbors(x, y)
        while t < max_iter:
            if t % reset == 0:
                active_set = self._get_active_set(x, y, neighbors)
                last_error = np.inf
                #lr = _lr # reset lr: 

            # recovery M
            temp = M.get_value()

            res = np.array(self.train_model(active_set, neighbors, lr)).flatten()
            error = res.sum()
            if self.verbose:
                print 'Iter: {} lr: {} pull_error: {} push_error: {} '.format(t, lr, res[0], res[1])

            if last_error>=error:
                self.theano_SPDproject()
                lr *= 1.01
            else:
                M.set_value(temp)
                lr *= 0.5
            # lr = lr*1.01*(last_error>error) + lr*0.5*(last_error<=error) 

            last_error = error
            t += 1

        self.trained = True
        if self.autosave: self.save('LMNN.model')
        return self

    def save(self, filename):
        cPickle.dump((self.__class__, self.__dict__), open(filename, 'w'))

    @staticmethod
    def load(filename):
        cls, attributes = cPickle.load(open(filename, 'r'))
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj

    @property
    def L(self):
        if self.trained:
            return __LDL__(self.M.get_value(), combined=True)
        else:
            raise Exception("Model hasn't trained yet!")

    def transform(self, x):
        return x.dot(self.L)

    def __theano_SPDproject(self, mat):
        # force symmetric
        mat = (mat+mat.T)/2.0
        eig, eigv = linalg.eig(mat)
        eig = T.maximum(eig, 0)
        eig = T.diag(eig)
        return eigv.dot(eig).dot(eigv.T) 

    def _get_active_set(self, x, y, neighbors):
        result = []
        ijcandidates = neighbors[np.random.choice(range(neighbors.shape[0]), size=(self.maxS))]
        lcandidates = np.random.randint(0, x.shape[0], size=(self.maxS, 1) )
        ijl = np.hstack([ijcandidates, lcandidates])
        return np.array(ijl, dtype='int32')

    def _get_neighbors(self, x, y):
        # shared neighbor
        n = x.shape[0]
        x = x.reshape(n, -1)
        neighbors = np.zeros((n, self.k), dtype=int)
        yset = np.unique(y)
        COST = 0.0
        for c in yset:
            mask = y==c
            ind = np.arange(n)[mask]
            _x = x[mask]

            for i in xrange(n):
                if y[i] != c: continue
                v = x[i] - _x
                cost = (v**2).sum(1)#np.diag(v.dot(self._M).dot(v.T))
                neighbors[i] = ind[cost.argsort()[1:self.k+1]]
                COST += sum(sorted(cost)[1:self.k+1])
        print 'neighbor cost: {}'.format(COST)

        return np.array(np.vstack([np.repeat(np.arange(x.shape[0]), self.k), neighbors.flatten()]).T, dtype='int32')

if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf, precision=3)

    from sklearn.datasets import load_digits
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cross_validation import train_test_split
    from KNN import __knn
    digits_data = load_digits()
    x = digits_data['data']
    y = digits_data['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.60)

    lmnn = LMNN(k=5, mu=1-1e-2, verbose=True)
    L = lmnn.fit(x_train, y_train, maxS=1000, lr=1e-5, max_iter=500, reset=20).L
    print L.shape

    def knn(x_train, x_test, y_train, y_test):
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(x_train, y_train)
        print float((neigh.predict(x_test) == y_test).sum())/y_test.shape[0]

    L[np.isnan(L)] = 0
    print __knn(x_train, x_test, y_train, y_test, K=5)
    print __knn(x_train, x_test, y_train, y_test, M=lmnn.M.get_value(), K=5)


