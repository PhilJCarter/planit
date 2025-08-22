"""
   planit rho-U / HM80 table wrapper functions
"""

from ..main import *
from .eos_table import *
from .eostab_extension import *
import numpy as npy


def loadrhoUEOS(eos='HM80-HHe-v2.0', eostype='HM80', debug = False):
    """
    READ IN rhoU tabulated EOS and fill the extEOStable class object
    
    Returns: EOStable
    """
    
    NewEOS  = EOStable() # make new empty EOS object
        
    if eos == 'HM80-HHe-v2.0':
        eosdir = eospath + 'HM80_HHe/'
        NewEOS.VERSION = 2.0
        NewEOS.loadrhoUtable(eosdir+'HM80_HHe_extended.txt')
    
    NewEOS.TYPE = eostype
    
    NewEOS.MODELNAME = eos

    return NewEOS

