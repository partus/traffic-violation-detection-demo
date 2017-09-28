#
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
#
from numpy import *
from numpy import linalg
import numpy as np
from scipy.sparse.csgraph import connected_components
import itertools
def perp( a ) :
    b = empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

import matplotlib.pyplot as plt
%matplotlib inline
# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = dot( dap, db)
    num = dot( dap, dp )
    coef = (num / denom.astype(float))
    if(coef > 1):
        return None
    else:
        return coef*db + b1

def smallAngled(a,b):
    crit = 1- math.cos(0.04)
    return abs(np.dot(a,b)/ linalg.norm(a)/linalg.norm(b)) > crit

def lineToVec(l):
    pair = np.array(l[0])
    a = pair[0:2]
    b = pair[2:4]
    return a,b

def areClose(l1,l2):
    a1,a2 = lineToVec(l1)
    b1,b2 = lineToVec(l2)
    ad =  a2-a1
    bd =  b2-b1
    a,b = linalg.norm(ad),linalg.norm(bd)
    con = ((a1+a2) -(b1+b2))/2
    connorm = linalg.norm(con)
    if(1.1* connorm > (a+b)/2):
        return False
    if(smallAngled(ad,bd) and smallAngled(ad,con) ):
        return True
    else:
        return False


def joinClose(lines):
    graph = np.zeros((lines.shape[0],lines.shape[0]))
    for line in lines:
    gr = []
    num = 0
    for line in lines:
        matches = [el for el in gr if areClose(el[0],line)]
        pair = np.array(line[0]).astype('uint32')
        a = pair[0:2]
        b = pair[2:4]
        a[0] += slip[0]
        a[1] += slip[1]
        b[0] += slip[0]
        b[1] += slip[1]
        vec = a - b

lines[1,...]
lines = hlines
lines.shape
def asdf(x,y):
    print(x,y)
    return x,y
    return areClose(lines[x,...],lines[y,...])
graph = np.zeros((lines.shape[0],lines.shape[0]), dtype=np.bool)
np.fromfunction(asdf, (lines.shape[0],lines.shape[0]))
hlines.shape
it = itertools.permutations(itertools.islice(itertools.count(),lines.shape[0]),2)
for x,y in it:
    graph[x,y] = areClose(lines[x,...],lines[y,...])
plt.hist(graph.ravel())
connected_components(graph, directed=False)
