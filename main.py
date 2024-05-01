### Module for accessing gadget data
###

"""Classes and functions for accessing and manipulating Gadget data."""

import numpy as npy
import scipy
import matplotlib
import matplotlib.pyplot as plt
import numba
import struct
import glob
import os
import inspect
import h5py
import woma

eospath = os.path.expanduser('~') + '/Work/'

from makeplanets import eostable
from .eos import EOStable, loadEOS
#from .eos import eospath

GADGET_EOS_OFFSET = IDOFF = 200000000    # material id offset
PROJ_ID_OFFSET = BODYOFF = 100000000     # body id offset




@numba.njit(parallel=True)
def calc_potential(m,x,y,z):
    G = 6.67e-8
    pot = npy.zeros(len(m))
    for j in numba.prange(len(m)):
        pdist = npy.sqrt( (x-x[j])**2 + (y-y[j])**2 + (z-z[j])**2 )
        pot[j] = npy.nansum(npy.where( npy.isinf(-G * m / pdist),0,-G * m / pdist ))
    return pot


IronEOS = loadEOS(eos='Iron-ANEOS-SLVTv0.2G1')
AlloyEOS = loadEOS(eos='Fe85Si15-ANEOS-SLVTv0.2G1')
ForsteriteEOS = loadEOS(eos='Forsterite-ANEOS-SLVTv1.0G1')




if __name__ == "__main__":
    import sys

    file = 'snapshot_000'
    if len(sys.argv) > 1:
        file = sys.argv[1]

    snap0 = Snapshot()
    print(snap0.header.flag_entr_ics)

    snap0.load(file, thermo=False)

    print(snap0.header.flag_entr_ics, snap0.header.flag_metals)
    #print file, snap0.U.max()/1.e11, 0.5*npy.sqrt(snap0.vx**2+snap0.vy**2+snap0.vx**2).max()


    print(snap0.N)
    #print snap0.header.num_files

    #print snap0.id[:5], snap0.id[-5:], snap1.id[:5], snap1.id[-5:]
    #print snap0.m[:5], snap0.m[-5:], snap1.m[:5], snap1.m[-5:]
    #print snap0.x[:5], snap0.x[-5:], snap1.x[:5], snap1.x[-5:]
    #print snap0.vx[:5], snap0.vx[-5:]
    
    
    #snap0.identify()

    #plt.scatter(snap0.x[snap0.fors == 1], snap0.y[snap0.fors == 1], c='b')
    #plt.scatter(snap0.x[snap0.iron == 1], snap0.y[snap0.iron == 1], c='orange')
    #plt.show()
