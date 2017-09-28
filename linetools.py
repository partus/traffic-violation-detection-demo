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
    if(connorm > 1.2*(a+b)/2):
        return False
    if(smallAngled(ad,bd) and smallAngled(ad,con) ):
        return True
    else:
        return False
def joinTwoLines(l1,l2):
    maxnorm = 0
    pair = ()
    for i in (0,2):
        for j in (0,2):
            p1,p2 = l1[0,i:i+2], l2[0,j:j+2]
            norm = linalg.norm(p1-p2)
            if norm  > maxnorm:
                pair = (p1,p2)
                maxnorm = norm
    return np.array(pair).reshape((1,4))

def joinClose(lines):
    graph = np.zeros((lines.shape[0],lines.shape[0]))
    it = itertools.permutations(itertools.islice(itertools.count(),lines.shape[0]),2)
    for x,y in it:
        graph[x,y] = areClose(lines[x,...],lines[y,...])
    labelcount, labels = connected_components(graph, directed=False)
    print(labels)
    lres = np.zeros((labelcount,1,4))
    for label in range(labelcount):
        lns = [lines[i] for i in range(len(labels)) if labels[i] == label ]
        line = lns[0]
        for ln in lns:
            line = joinTwoLines(line,ln)
        lres[label,...] = line
    return lres


# hlines
# joinClose(hlines)
