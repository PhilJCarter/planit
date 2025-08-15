"""
   planit AQUA table wrapper functions
"""

from ..main import *
from .eos_table import *
from .eostab_extension import *
import numpy as npy


def loadAQUAEOS(eos='Water-AQUA-v1.0', eostype='AQUA', debug = False):
    """
    READ IN AQUA EOS and fill the extEOStable class object
    
    Returns: EOStable
    """
        
    if eos == 'Water-AQUA-v1.0':
        eosdir = eospath + 'aqua-water/'

    
    NewEOS  = EOStable() # FIRST make new empty EOS object
    NewEOS.TYPE = eostype
    NewEOS.VERSION = 1.0
    
    NewEOS.loadaquatable(eosdir+'aqua_eos_rhot_v1_0.dat')
    
    NewEOS.MODELNAME = eos # string set above in user input
    NewEOS.MDQ = npy.zeros((NewEOS.NT,NewEOS.ND)) # makes the empty MDQ array

    
#    NewEOS.calchugoniot(r0=NewEOS.R0REF,t0=NewEOS.T0REF)
    #
    # calculate the 1-bar profile; loop over temp
#    NewEOS.onebar.T = npy.zeros(NewEOS.NT)
#    NewEOS.onebar.S = npy.zeros(NewEOS.NT)
#    NewEOS.onebar.rho = npy.zeros(NewEOS.NT)
#    it0 = npy.where(NewEOS.T >= NewEOS.T0REF)[0]
#    id0 = npy.arange(NewEOS.ND)#npy.where(NewEOS.rho >= 0.8*NewEOS.R0REF)[0]
#    for iit in range(0,NewEOS.NT):
#        NewEOS.onebar.T[iit] = NewEOS.T[iit]
#        NewEOS.onebar.S[iit] = npy.interp(1.E-4,NewEOS.P[iit,id0],NewEOS.S[iit,id0])
#        NewEOS.onebar.rho[iit] = npy.interp(1.E-4,NewEOS.P[iit,id0],NewEOS.rho[id0])

    return NewEOS

