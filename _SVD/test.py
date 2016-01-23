import numpy as np
import cProfile
#import pyximport; pyximport.install()
 
N = 800
a = np.random.uniform(0, 100, size=(N, N))

import pure_numpy
print '='*100
print '\tpure python'
cProfile.run('pure_numpy.SVD(a)')

import bytheano 
print '='*100
print '\ttheano nlinalg.svd'
cProfile.run('bytheano.SVD(a)')


