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
def normalize(a):
    return a/linalg.norm(a)
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

def seg_poliline_intersect(seg,pol):
    res = False
    for l1,l2 in itertools.izip(pol,pol[1:]):
        if not seg_intersect(l1[0:2],l1[2:4],l2[0:2],l2[2:4]) is None:
            res=True
            break
    return res


def smallAngled(a,b):
    # return False
    crit = math.cos(0.05)
    return abs(np.dot(a,b)/ linalg.norm(a)/linalg.norm(b)) > crit

def lineToVec(l):
    pair = l[0,...]
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
    if(smallAngled(ad,bd) and (abs(np.dot(normalize(perp(ad)),con)) < 5)):
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
    lres = np.zeros((labelcount,1,4))
    glines =[]
    for label in range(labelcount):
        tp = [labels[i] for i in range(len(labels)) if labels[i] == label ]
        lns = [lines[i] for i in range(len(labels)) if labels[i] == label ]
        line = lns[0]
        for ln in lns:
            line = joinTwoLines(line,ln)
        glines.append(np.array(lns))
        lres[label,...] = line
    # return glines
    return lres


# hlines
# areClose(np.array([[0,0,100,0]]),np.array([[0,0,60,80]]))
# joinClose(hlines)
