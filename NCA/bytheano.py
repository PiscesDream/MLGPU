import numpy as np
import theano
import theano.tensor as T
import sys

class NCA(object):

    def __init__(self, dim=-1):
        self.dim = dim 
        self.__EPSILON__ = 1e-10

    def __Theano_build__(self):
        Tx = T.matrix('Tx')
        Ty = T.ivector('Ty')
        Tlr = T.scalar('Tlr')
        if self.dim != self.X.shape[1]:
            A = theano.shared(name='A', value=np.random.rand(self.X.shape[1], self.dim))
        else:
            A = theano.shared(name='A', value=np.eye(self.dim))

        # sequences (if any), prior result(s) (if needed), non-sequences (if any)
        def fn(x, y, trainx, trainy):
            # to avoid nan (when dist -> inf, exp(-dist) -> 0.0,  xx / 0.0 -> nan or inf)
            # method 1 epsilon
            #   prob = prob / (prob.sum()-1+self.__EPSILON__)
            # method 2 backtracking
            prob = T.exp(-((x - trainx) ** 2).sum(1))
            prob = prob / (prob.sum()-1)
            prob = T.switch(T.gt(prob, 1), 0, prob)
            correctness = T.switch(T.eq(y, trainy), prob, 0)
            return correctness
#            return [correctness, debug]

        correctness, _ = theano.scan(fn=fn, 
#        [correctness, debug], _ = theano.scan(fn=fn, 
            outputs_info=None, 
            sequences=[Tx.dot(A), Ty], 
            non_sequences=[Tx.dot(A), Ty])
# second way to calc pij
#       pij = dists/(dists.sum(1)-1).dimshuffle(0, 'x')
#       pij = T.switch(T.gt(pij, 1), 0, pij) # * (1-T.eye(self.X.shape[0]))

        all_correctness = T.sum(correctness)
        grad = T.grad(all_correctness, A)
        updates = [(A, A + Tlr*grad)]

        self.Ttrain = theano.function([Tlr], 
                [],
#                [debug, all_correctness],
                givens={Tx: self.X, 
                        Ty: self.y},
                updates=updates,
                allow_input_downcast=True,
                on_unused_input='warn')

        self.Tcorrectness = theano.function([], 
                all_correctness,
                givens={Tx: self.X, Ty: self.y},
                allow_input_downcast=True)

        self.Ttransform = theano.function([Tx], Tx.dot(A), allow_input_downcast=True)
        self.TA = A

    def fit(self, X, y, max_iter=1000, lr=1e-3):
        self.X = np.array(X, dtype=theano.config.floatX)
        self.y = np.array(y, dtype='int32')
        if self.dim == -1:
            self.dim = X.shape[1]
        self.classes = len(set(y))
        self.__Theano_build__()
        
        last_correctness = 0
        for i in range(max_iter):
            lastA = self.A
            self.Ttrain(lr)
            correctness = self.Tcorrectness()
            print 'Iter[{}]: lr={}, f(A)={}'.format(i, lr, correctness)
            if correctness < last_correctness or np.isnan(correctness): 
                lr *= 0.5 
                self.A = lastA # backtrack lr and A
            else:
                lr *= 1.01
                last_correctness = correctness
            sys.stdout.flush()
        return self
        
    def transform(self, X):
        return self.Ttransform(X)

    @property
    def A(self):
        return self.TA.get_value()

    @A.setter
    def A(self, value):
        self.TA.set_value(value)

if __name__ == '__main__':
#   np.set_printoptions(linewidth=100)
#   print "Test 1, simple test"
#   X = np.array([[0, 0], [1, 1], [2, 0], [3, 1]], dtype='float32')
#   y = np.array([0, 0, 1, 1], dtype='int32')
#   nca = NCA()
#   nca.fit(X, y, max_iter=10, lr=100)
#   print nca.A

    import sys; sys.path.append('/home/shaofan/Projects') 
    from FastML import KNN
    print "\nTest 3, LFW"
    from sklearn.datasets import fetch_lfw_people
    from sklearn.decomposition import PCA 
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import normalize
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    pca = PCA(n_components=100)
    X = pca.fit_transform(lfw_people.data)
    X = normalize(X)
    y = lfw_people.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    knn = KNN(n_neighbors=4)
    pred = knn.fit(X_train, y_train).predict(X_test)
    print '{}/{}'.format( (pred == y_test).sum(), len(y_test) )

    nca = NCA()
    X_train = nca.fit(X_train, y_train, max_iter=10, lr=5e-3).transform(X_train)
    X_test = nca.transform(X_test)

    knn = KNN(n_neighbors=4)
    pred = knn.fit(X_train, y_train).predict(X_test)
    print '{}/{}'.format( (pred == y_test).sum(), len(y_test) )


