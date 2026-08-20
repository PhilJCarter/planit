"""
planit  (PLANetary Impact Toolkit)
======

Classes and functions for setting-up, accessing, manipulating, 
and analysing SPH impact simulation data.

Supported planetary impact codes: gadget-2-planetary, swift


Philip J. Carter (p.carter@bristol.ac.uk)

Sarah T. Stewart (sstewa56@asu.edu)

v1.0

"""

from .globaldefs import *
from . import eos
from . import utils
from .snaptools import Snapshot, io
from .impacttools import Impact


IronEOS       = eos.ANEOSIron        # for backwards compatibility
AlloyEOS      = eos.ANEOSFeSiAlloy
ForsteriteEOS = eos.ANEOSForsterite


def loadsnap(file, headonly = False, thermo=False, compress=False, mats=[402,400]):
    s = Snapshot()
    s.load(file, headonly=headonly, thermo=thermo, compress=compress, mats=mats)
    return s

def loadimpact(loc,thermo=False,inter=1,compress=True,code='swift'):
    i = Impact()
    i.load(loc, thermo=thermo, inter=inter, compress=compress, code=code)
    return i
    
def load_seagen(partplanet, thermo=False, init_h=100e5):
    s = Snapshot()
    io.load_seagen(s, partplanet, thermo=thermo, init_h=init_h)
    return s

def combine(body1, body2, bidoffset=PROJ_ID_OFFSET, thermo=False, box=0.0):
    s = Snapshot()
    s.combine(body1, body2, bidoffset=bidoffset, thermo=thermo, box=box)
    return s

def bound_mass(snapshot, nrem = 1, minbnd = 200, maxiter = 2000, tol = 0.01, reorder=True, discardsmall=False, calc_pot_all=True, save=True):
    snapshot.bound_mass(nrem = nrem, minbnd = minbnd, maxiter = maxiter, tol = tol, reorder=reorder, discardsmall=discardsmall, calc_pot_all=calc_pot_all, save=save)
    return snapshot.rem

def calc_phase(snapshot, release=False, plot=False):
    snapshot.calc_phase(release=release, plot=plot)
    return snapshot.phase
