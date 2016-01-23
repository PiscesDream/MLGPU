import numpy as np
import cProfile
#import pyximport; pyximport.install()

N = 300
a = np.random.uniform(0, 100, size=(N, N))

import pure_python 
print '='*100
print '\tpure python'
cProfile.run('pure_python.LDL(a)')

import ldl1
print '='*100
print '\tCython on loop'
cProfile.run('ldl1.LDL(a)')

import ldl2
print '='*100
print '\twithout bounds check + [dtype, ndim]'
cProfile.run('ldl2.LDL(a)')

