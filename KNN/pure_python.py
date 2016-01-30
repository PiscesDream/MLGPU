import numpy as np

class KNN(object):

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.X = self.y = None

    def fit(self, X, y):
        self.X = np.array(X).reshape(len(X), -1) # flattern
        self.y = np.array(y).flatten()
        return self

    def predict(self, testx):
        if self.X == None: raise Exception("KNN no trained yet !")

        testx = np.array(testx).reshape(len(testx), -1) # flatten
        assert (self.X.shape[1] == testx.shape[1])
        prediction = []
        for x in testx:
            dist = ((x - self.X) ** 2).sum(1)
            candidates = self.y[dist.argsort()[:self.n_neighbors]]
            b = np.bincount(candidates)
            prediction.append( y[b.argmax()] )
        return np.array(prediction)

    def predict2(self, testx):
        if self.X == None: raise Exception("KNN no trained yet !")

        testx = np.array(testx).reshape(len(testx), -1) # flatten
        assert (self.X.shape[1] == testx.shape[1])
        prediction = []
        candidates = self.y[ ((np.tile(testx, (self.X.shape[0], 1, 1)).swapaxes(0, 1) - self.X) ** 2).sum(2).argsort()[:, :self.n_neighbors] ]
        for c in candidates:
            b = np.bincount(c)
            prediction.append( y[b.argmax()] )
        return np.array(prediction)

    def predict_proba(self, testx):
        if self.X == None: raise Exception("KNN no trained yet !")

        testx = np.array(testx).reshape(len(testx), -1) # flatten
        assert (self.X.shape[1] == testx.shape[1])
        proba = []
        for x in testx:
            dist = ((x - self.X) ** 2).sum(1)
            candidates = self.y[dist.argsort()[:self.n_neighbors]]
            b = np.bincount(candidates)
            proba.append( b.astype(float)/self.n_neighbors )
        return np.array(proba)

if __name__ == '__main__':
    print "Test 1, simple test"
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    neigh = KNN(n_neighbors=3)
    neigh.fit(X, y) 
    print(neigh.predict([[1.1]]))
    print(neigh.predict_proba([[0.9]]))

    print "\nTest 2, digit test"
    from sklearn.datasets import load_digits
    from sklearn.cross_validation import train_test_split
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    knn = KNN(n_neighbors=5)
    pred = knn.fit(X_train, y_train).predict(X_test)
    print '{}/{}'.format( (pred == y_test).sum(), len(y_test) )


    import cProfile
#   cProfile.run('knn.predict(X_test)') # FASTER 
#   cProfile.run('knn.predict2(X_test)')



    print "\nTest 3, LFW"
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
#    cProfile.run('knn.fit(X_train, y_train).predict(X_test)') 
