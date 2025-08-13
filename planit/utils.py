from .main import *
import numpy as npy
import numba


class Node:
    def __init__(self,width,origin,pos,mass,ids,leaves=[]):
        self.size = width
        self.origin = origin
        self.children = []
    
        N = len(mass)
        if N==1:
            self.m = mass[0]
            self.com = pos[0]
            self.id = ids[0]
            self.pot = 0
            leaves.append(self)
        elif N==2 and npy.sqrt(((pos[0]-pos[1])**2).sum())<(self.size/2e50):
            self.m = mass.sum()
            self.com = (pos[0]+pos[1])/2.
            self.id = ids[0]
            self.pot = 0
            leaves.append(self)
        elif self.size == 0.0:
            if len(ids)>2:
                print('WARNING: ', len(ids), ' overlapping particles: ',ids)
            self.m = mass.sum()
            self.com = pos[0]
            self.id = ids[0]
            self.pot = 0
            leaves.append(self)
        else:
            self.genChild(width,origin,pos,mass,ids,leaves)
        
            m = [c.m for c in self.children]
            com = [c.com for c in self.children]
            m = npy.array(m)
            com = npy.array(com)
            self.m = m.sum()
            self.com = (m*com.T).T.sum(axis=0)/self.m
    
    
    def genChild(self,width,origin,pos,mass,ids,leaves):
        oct_choose = pos > self.origin
        oct_choose = oct_choose.astype(int)
        for i in [0,1]:
            for j in [0,1]:
                for k in[0,1]:
                    this_oct = (oct_choose[:,0] == i)*(oct_choose[:,1] == j)*(oct_choose[:,2] == k)
                    if npy.any(this_oct):
                        dw = 0.5*self.size*(npy.array([i,j,k])-0.5)
                        self.children.append(Node(0.5*self.size,self.origin+dw,pos[this_oct],mass[this_oct],ids[this_oct],leaves))



def Walk_Pot(node,leaf,theta2):
    dr = node.com - leaf.com
    r2 = (dr**2).sum()
    if (len(node.children)==0) or (node.size**2/r2 < theta2):
        if r2!=0:
            leaf.pot += - G * node.m / npy.sqrt(r2)
    else:
        for c in node.children:
            Walk_Pot(c,leaf,theta2)
            


@numba.njit(parallel=True)
def _calc_potential_direct(m,x,y,z):
    pot = npy.zeros(len(m))
    for j in numba.prange(len(m)):
        pdist = npy.sqrt( (x-x[j])**2 + (y-y[j])**2 + (z-z[j])**2 )
        pot[j] = npy.nansum(npy.where( npy.isinf(-G * m / pdist),0,-G * m / pdist ))
    return pot


def _calc_potential_tree(m,x,y,z):
    theta = 1.1
    theta2 = theta**2
    treeorigin = (npy.max((x,y,z),axis=1) + npy.min((x,y,z),axis=1))/2.
    treewidth = (npy.max((x,y,z)) - npy.min((x,y,z)))
    leaves = []
    tree = Node(treewidth,treeorigin,npy.array((x,y,z)).T,m,npy.arange(len(m)),leaves)
    
    pot = npy.zeros(len(m))
    for leaf in leaves:
        Walk_Pot(tree,leaf,theta2)
        pot[leaf.id] = leaf.pot
    return pot


def calc_potential(m,x,y,z):
    if len(m) > 200000:
        try:
            return _calc_potential_tree(m,x,y,z)
        except:
            return _calc_potential_direct(m,x,y,z)
    else:
        return _calc_potential_direct(m,x,y,z)
    
