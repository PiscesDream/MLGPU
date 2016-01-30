import numpy as np
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

class MCML(object):

    def __init__(self, lmbd=1.0):
        self.lmbd = lmbd

    def __Theano_build__(self):
        Tx = T.matrix('Tx')
        Ty = T.ivector('Ty')
        Tlr = T.scalar('Tlr')
        A = theano.shared(name='A', value=np.eye(self.X.shape[1]))

        def fn1(x, y, trainx, trainy):
            diff = x - trainx
            cost = linalg.diag(diff.dot(A).dot(diff.T)) * T.eq(y, trainy)
            return cost

        def fn1_2(i, j, x, y):
            diff = x[i] - x[j] 
            cost = T.eq(y[i], y[j]) * diff.dot(A).dot(diff.T)
            return cost

        # sequences (if any), prior result(s) (if needed), non-sequences (if any)
        def fn2(x, trainx):
            cost = T.exp(-((x - trainx) ** 2).sum(1))
            return cost.sum() 

        n = self.X.shape[0]
        indices = np.meshgrid(np.arange(n), np.arange(n))
        iind = theano.shared(name='iind', value=indices[0].flatten())
        jind = theano.shared(name='jind', value=indices[1].flatten())
        costs1, _ = theano.scan(fn=fn1_2, outputs_info=None, sequences=[iind, jind], non_sequences=[Tx, Ty])

#       costs1, _ = theano.scan(fn=fn1, outputs_info=None, sequences=[Tx, Ty], non_sequences=[Tx, Ty])
        costs2, _ = theano.scan(fn=fn2, outputs_info=None, sequences=[Tx], non_sequences=[Tx])

        loss = T.sum(costs1) + self.lmbd * T.sum(costs2)
        grad = T.grad(loss, A)
        updates = [(A, SPD_Project(A - Tlr*grad))]

        self.Ttrain = theano.function([Tlr], 
                [],
                givens={Tx: self.X, 
                        Ty: self.y},
                updates=updates,
                allow_input_downcast=True,
                on_unused_input='warn')

        self.Tloss = theano.function([], 
                loss,
                givens={Tx: self.X, Ty: self.y},
                allow_input_downcast=True)

        self.Ttransform = theano.function([Tx], Tx.dot(A), allow_input_downcast=True)
        self.TA = A

    def fit(self, X, y, max_iter=1000, lr=1e-3):
        self.X = np.array(X, dtype=theano.config.floatX)
        self.y = np.array(y, dtype='int32')
        self.dim = X.shape[1]
        self.classes = len(set(y))
        self.__Theano_build__()
        
        from datetime import datetime
        print 'Initial loss: {}'.format(self.Tloss())
        last_loss = np.inf
        for i in range(max_iter):
            lastA = self.A
            startt = datetime.now()
            self.Ttrain(lr)
            print 'Training use:', (datetime.now() - startt).total_seconds()
            loss = self.Tloss()
            print 'Iter[{}]: lr={}, L(A)={}'.format(i, lr, loss)
            if loss > last_loss or np.isnan(loss): 
                lr *= 0.5 
                self.A = lastA # backtrack lr and A
            else:
                lr *= 1.01
                last_loss = loss
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
#   mcml = MCML()
#   mcml.fit(X, y, max_iter=10, lr=0.1)
#   print mcml.A
#   print mcml.transform(X)

    import sys; sys.path.append('/home/shaofan/Projects') 
    from FastML import KNN
    print "\nTest 3, LFW"
    from sklearn.datasets import fetch_lfw_people
    from sklearn.decomposition import PCA 
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import normalize
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    pca = PCA(n_components=10)
    X = pca.fit_transform(lfw_people.data)
    X = normalize(X)
    y = lfw_people.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    knn = KNN(n_neighbors=2)
    pred = knn.fit(X_train, y_train).predict(X_test)
    print '{}/{}'.format( (pred == y_test).sum(), len(y_test) )

    knn = KNN(n_neighbors=2)
    mcml = MCML()
    for _ in range(100):
        X_train = mcml.fit(X_train, y_train, max_iter=5, lr=1e-6).transform(X_train)
        X_test = mcml.transform(X_test)

        pred = knn.fit(X_train, y_train).predict(X_test)
        print '{}/{}'.format( (pred == y_test).sum(), len(y_test) )


