from numpy import *
from numpy import linalg
import numpy as np
from scipy.sparse.csgraph import connected_components
import itertools
import cv2
import copy

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
    numa = dot( dap, dp )
    coef = (numa / denom.astype(float))

    # dbp = perp(db)
    # numb = dot( dbp, dp )
    # coefb = (numb / denom.astype(float))
    #
    #id

    if(coef > 1 or coef < 0):
        return None
    else:
        return coef*db + b1

def seg_poliline_intersect(seg,pol):
    res = []
    for p1, p2 in zip(pol,pol[1:]):
        # print(p1,p2)
        point = seg_intersect(seg[0,0:2],seg[0,2:4],p1[0],p2[0])
        pointr = seg_intersect(p1[0],p2[0],seg[0,0:2],seg[0,2:4])
        if not point is None and not pointr is None:
            res.append(point)
            # break
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
    if(smallAngled(ad,bd) and (abs(np.dot(normalize(perp(ad)),con)) < 15)):
        return True
    else:
        return False

areSame = areClose

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

def draw_lines(img, lines, extend=0, color=[255, 0, 0],extendColor=[0, 255, 0], thickness=1, slip=(0, 0)):
    if not lines is None:
        for line in lines:
            pair = np.array(line[0]).astype('uint32')
            a = pair[0:2]
            b = pair[2:4]
            a[0] += slip[0]
            a[1] += slip[1]
            b[0] += slip[0]
            b[1] += slip[1]
            vec = a - b
            if extend:
                cv2.line(img, tuple(a + vec * extend),
                         tuple(b - vec * extend),extendColor , thickness)
            cv2.line(img, tuple(a), tuple(b), color, thickness)
        # return img
class LineStorage:
    def __init__(self):
        self.lines = []
        self.groups = {}
        True

    def apply(self,lines):
        self.lines = lines
        for id,group in self.groups.items():
            group['other'] = []
            for line in lines:
                if(areClose(group['main'],line)):
                    group['other'].append(line)
        True

    def clickMatch(self,point):
        print(self.lines)
        print("POINT",point)
        for line in self.lines:
            a,b = lineToVec(line)
            if(linalg.norm(a-point) + linalg.norm(b-point) < linalg.norm(a-b)+1):
                for id,group in self.groups.items():
                    if(areClose(line,group['main'])):
                        group['main'] = line
                        return id
                id = self.getSmallest()
                self.groups[id] = {
                    # type - Neitral, Parallel, Red,Green,Yello
                    'type': 'P',
                    'main': line,
                    'other': []
                }
                return id
        return None

    def getGroupsCopy(self):
        return copy.deepcopy(self.groups)
    def getGroups(self):
        return self.groups
    def setType(self,id, type):
        self.groups[id]['type']=type
    def remove(self,id):
        if id in self.groups:
            del self.groups[id]
        True
    def onTypeChoose(self,id,name):
        self.groups[id]['type'] = name
    def getSmallest(self):
        i= 0
        while i in self.groups:
            i+=1
        return i

# hlines
# areClose(np.array([[0,0,100,0]]),np.array([[0,0,60,80]]))
# joinClose(hlines)
