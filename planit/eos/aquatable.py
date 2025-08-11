from ..main import *
from .eos_table import *
import numpy as npy


class EOStable(extEOStable):
    def __init__(self):
        extEOStable.__init__(self)
        self.TYPE = ''
        #self.name = ''
        
    def loadaquatable(self,fname):
        with open(fname,'r') as tablefile:
            ND = None
            NT = None
            while(ND is None or NT is None):
                tmp =tablefile.readline()
                if tmp.count('rho')>0:
                    ND = int(tmp.split('(')[1].split()[0])
                if tmp.count('temp ')>0:
                    NT = int(tmp.split('(')[1].split()[0])
        self.ND = ND
        self.NT = NT
        rho, T, P, S, U, cs, phase = npy.loadtxt(fname,skiprows=21,usecols=(0,1,2,4,5,6,10),unpack=True)
        self.T = T[0:self.NT]
        self.rho = rho[::self.NT]/1000.
        self.P = P.reshape(self.ND,self.NT).T/1.e9
        self.S = S.reshape(self.ND,self.NT).T/1.e6
        self.U = U.reshape(self.ND,self.NT).T/1.e6
        self.cs = cs.reshape(self.ND,self.NT).T*100.
        phase = npy.where(phase == 3,7,phase)
        phase = npy.where(phase == 5,8,phase)
        phase = npy.where(phase == 4,6,phase)
        phase = npy.where(phase == 2,5,phase)
        phase = npy.where(phase == 0,2,phase)
        phase = npy.where(phase == 1,2,phase)
        phase = npy.where(phase < 0,4,phase)
        self.KPA = phase.reshape(self.ND,self.NT).T



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

