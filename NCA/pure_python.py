import numpy as np

# from http://all-umass.github.io/metric-learn/_modules/metric_learn/nca.html#NCA.fit
class NCA(object):
    
    def __init__(self, max_iter=100, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.A = None

    def transform(self, X):
        return np.dot(X, self.A)

    def fit_transform(self, X, labels):
        return self.fit(X, labels).transform(X)

    def transformer(self):
        return self.A

    def fit(self, X, labels):
        n, d = X.shape
        A = np.zeros((d, d))
        np.fill_diagonal(A, 1./(X.max(axis=0)-X.min(axis=0)))

        # Run NCA
        dX = X[:,None] - X[None]  # shape (n, n, d)
        tmp = np.einsum('...i,...j->...ij', dX, dX)  # shape (n, n, d, d)
        masks = labels[:,None] == labels[None]
        for it in xrange(self.max_iter):
            print it
            for i, label in enumerate(labels):
                mask = masks[i]
                Ax = A.dot(X.T).T  # shape (n, d)

                softmax = np.exp(-((Ax[i] - Ax)**2).sum(axis=1))  # shape (n)
                softmax[i] = 0
                softmax /= softmax.sum()

                t = softmax[:, None, None] * tmp[i]  # shape (n, d, d)
                d = softmax[mask].sum() * t.sum(axis=0) - t[mask].sum(axis=0)
                A += self.learning_rate * A.dot(d)

        self.X = X
        self.A = A
        return self


if __name__ == '__main__':
    import sys; sys.path.append('/home/shaofan/Projects') 
    from FastML import KNN

#   print "Test 1, simple test"
#   X = [[0], [1], [2], [3]]
#   y = [0, 0, 1, 1]
#   neigh = KNN(n_neighbors=3)
#   neigh.fit(X, y) 
#   NCA.fit
#   print(neigh.predict([[1.1]]))
#   print(neigh.predict_proba([[0.9]]))

#   print "\nTest 2, digit test"
#   from sklearn.datasets import load_digits
    from sklearn.cross_validation import train_test_split
#   digits = load_digits()
#   X, y = digits.data, digits.target
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
#   knn = KNN(n_neighbors=5)
#   pred = knn.fit(X_train, y_train).predict(X_test)
#   print '{}/{}'.format( (pred == y_test).sum(), len(y_test) )


    print "\nTest 3, LFW"
    from sklearn.datasets import fetch_lfw_people
    from sklearn.decomposition import PCA 
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    pca = PCA(n_components=10)
    X = pca.fit_transform(lfw_people.data)
    y = lfw_people.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    knn = KNN(n_neighbors=5)
    pred = knn.fit(X_train, y_train).predict(X_test)
    print '{}/{}'.format( (pred == y_test).sum(), len(y_test) )

    nca = NCA()
    X_train = nca.fit(X_train, y_train).transform(X_train)
    X_test = nca.transform(X_test)

    knn = KNN(n_neighbors=5)
    pred = knn.fit(X_train, y_train).predict(X_test)
    print '{}/{}'.format( (pred == y_test).sum(), len(y_test) )

#    import cProfile
#    cProfile.run('knn.predict(X_test)')
