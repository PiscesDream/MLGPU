import theano
import theano.tensor as T
import numpy as np

class KNN(object):

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.X = self.y = None

    def fit(self, X, y):
        self.X = np.array(X, dtype=theano.config.floatX).reshape(len(X), -1) # flattern
        self.y = np.array(y, dtype='int32').flatten()
        self.classes = len(set(y))
        self.__Theano_build__()
        return self

    def __Theano_build__(self):
        Tx = T.matrix('Tx')
        Ty = T.ivector('Ty')
        Ttestx = T.matrix('Ttestx')

        # sequences (if any), prior result(s) (if needed), non-sequences (if any)
        def fn(testx, trainx, trainy):
            c = trainy[ ((testx - trainx) ** 2).sum(1).argsort()[:self.n_neighbors] ]
            b = T.extra_ops.bincount(c, minlength=self.classes)
            return b.argmax()

        dists, _ = theano.scan(fn=fn,
                outputs_info=None,
                sequences=[Ttestx],
                non_sequences=[Tx, Ty])

        self.Tcalc = theano.function([Ttestx], 
                dists, 
                givens={Tx: self.X, 
                        Ty: self.y},
                allow_input_downcast=True)


    def predict(self, testx):
        if self.X == None: raise Exception("KNN no trained yet !")

        testx = np.array(testx).reshape(len(testx), -1) # flatten
        assert (self.X.shape[1] == testx.shape[1])
        return self.Tcalc(testx)

    def predict_proba(self, testx):
        pass

if __name__ == '__main__':
    print "Test 1, simple test"
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 2]], dtype='float32')
    y = np.array([0, 0, 1, 1], dtype='int32')
    neigh = KNN(n_neighbors=3)
    neigh.fit(X, y) 


    print "\nTest 2, digit test"
    from sklearn.datasets import load_digits
    from sklearn.cross_validation import train_test_split
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    knn = KNN(n_neighbors=5)
    pred = knn.fit(X_train, y_train).predict(X_test)
    print '{}/{}'.format( (pred == y_test).sum(), len(y_test) )


    print "\nTest 3, LFW"
    import cProfile
    from sklearn.datasets import fetch_lfw_people
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X = lfw_people.data
    y = lfw_people.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    knn = KNN(n_neighbors=5)
    pred = knn.fit(X_train, y_train).predict(X_test)
    print '{}/{}'.format( (pred == y_test).sum(), len(y_test) )
    cProfile.run('knn.predict(X_test)')
    #cProfile.run('knn.fit(X_train, y_train).predict(X_test)') 


