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
    
    NewEOS.MODELNAME = eos # string set above in user input
    NewEOS.MDQ = npy.zeros((NewEOS.NU,NewEOS.ND)) # makes the empty MDQ array

    
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

