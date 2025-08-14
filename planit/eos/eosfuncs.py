"""
   planit equation of state functions
"""

from ..main import *

import numpy as npy
import numba
from .eos_table import *
from .eos_table import isentrope_class as eos_isentrope_class
from .aneostable import *
from . import tabinterp


def loadEOS(eos='Iron-ANEOS-SLVTv0.2G1', eostype='ANEOS'):
    if eostype == 'ANEOS':
        return loadANEOSEOS(eos=eos, eostype='ANEOS')
    elif eostype == 'SESAME':
        return loadANEOSEOS(eos=eos, eostype='SESAME')
    else:
        raise ValueError('Error: unsupported EOS type:', eostype)


# Variables to hold EOS tables
ANEOSIron       = None
ANEOSFeSiAlloy  = None
ANEOSForsterite = None

# EOS table common names
ironnames  = ['iron','ANEOSIron','Fe','Iron',401]
alloynames = ['alloy','ANEOSFeSiAlloy','FeSi','Alloy','IronAlloy','ironalloy',402]
forsteritenames = ['forsterite','ANEOSForsterite','Forsterite','Fo',400]


def select(name):
    """
       Return EoS table object specified by name, 
       loading it first if not already loaded
    """
    if name in ironnames:
        global ANEOSIron
        if not ANEOSIron:
            ANEOSIron = loadEOS(eos='Iron-ANEOS-SLVTv0.2G1', eostype='ANEOS')
        return ANEOSIron
    elif name in alloynames:
        global ANEOSFeSiAlloy
        if not ANEOSFeSiAlloy:
            ANEOSFeSiAlloy  = loadEOS(eos='Fe85Si15-ANEOS-SLVTv0.2G1', eostype='ANEOS')
        return ANEOSFeSiAlloy
    elif name in forsteritenames:
        global ANEOSForsterite
        if not ANEOSForsterite:
            ANEOSForsterite = loadEOS(eos='Forsterite-ANEOS-SLVTv1.0G1', eostype='ANEOS')
        return ANEOSForsterite
    else:
        raise ValueError('Unknown EOS.')
        return None
        
        
class isentrope_class(eos_isentrope_class):
    """Class to hold isentrope data extracted from EOS table.
    
       extract(material,entropy) - extract isentrope from EOS
    
    """ 
    def __init__(self, entropy=None, material=None): 
        """A function to initialize the class object.""" 
        eos_isentrope_class.__init__(self)
        self.entropy = entropy
        self.material = material
        self.intenergy = []
        if self.material and self.entropy:
            self.extract()
    
    def extract(self,material=None,entropy=None):
        """Extract isentrope at entropy from EOS specified by material"""
        if  not self.material:
            if material:
                self.material = material
            else:
                print('error: no material specified')
                return
        if not self.entropy:
            if entropy:
                self.entropy = entropy
            else:
                print('error: entropy not specified')
                return
        EOS = eos.select(self.material)
        self.density = EOS.rho
        self.ND = EOS.ND
        
        # loop across all densities and extract the values for the requested isentrope
        for i in range(0,self.ND):
            ind = npy.where(EOS.S[:,i] > 0)[0]
            interpfunction = interpolate.interp1d(EOS.S[ind,i],EOS.P[ind,i]) # MJ/K/kg, GPa
            self.pressure = npy.append(self.pressure,interpfunction(self.entropy/1.E3)) # GPa
            interpfunction = interpolate.interp1d(EOS.S[ind,i],EOS.T[ind]) # MJ/K/kg, GPa
            self.temperature = npy.append(self.temperature,interpfunction(self.entropy/1.E3)) # GPa


# cgs / SI unit conversions
uconversion_m_cgs2SI = 1e-3
uconversion_l_cgs2SI = 1e-2
uconversion_v_cgs2SI = uconversion_m_cgs2SI
uconversion_rho_cgs2SI = uconversion_m_cgs2SI * uconversion_l_cgs2SI**-3
uconversion_S_cgs2SI = 1e-4
uconversion_P_cgs2SI = 1e-1
uconversion_U_cgs2SI = 1e-4

uconversion_m_SI2cgs = 1./uconversion_m_cgs2SI
uconversion_l_SI2cgs = 1./uconversion_l_cgs2SI
uconversion_v_SI2cgs = 1./uconversion_v_cgs2SI
uconversion_rho_SI2cgs = 1./uconversion_rho_cgs2SI
uconversion_S_SI2cgs = 1./uconversion_S_cgs2SI
uconversion_P_SI2cgs = 1./uconversion_P_cgs2SI
uconversion_U_SI2cgs = 1./uconversion_U_cgs2SI

# cgs / eos_table unit conversions
uconversion_S = uconversion_S_cgs2SI/1e6
uconversion_P = uconversion_P_cgs2SI/1e9
uconversion_U = uconversion_U_cgs2SI/1e6

uconversion_P_inv = 1./uconversion_P
uconversion_U_inv = 1./uconversion_U
uconversion_S_inv = 1./uconversion_S



#@numba.jit(parallel=True,forceobj=True)
def calcprop(Qlab,Xlab,Ylab,X,Y,mats):
    """Calculate thermodynamic property
       
       Qlab - label of property to calculate
       Xlab - label of 1st known property to calculate from
       Ylab - label of 2nd known property to calculate from
       X - array of 1st known property values
       X - array of 2nd known property values
       mats - array of material identifiers
       
       returns array of interpolated property at X, Y points
    """
    if npy.ndim(X) == 0:
        X = npy.array([X,])
        Y = npy.array([Y,])
        Z = npy.array([mats,])
    
    if not len(X)==len(Y)==len(mats):
        raise ValueError('X, Y, and mats arrays must be the same size/shape')
        
    Q = npy.zeros(len(X))
    if Ylab == 'rho' and Xlab in ['T','U','S']:
        tmp = Y
        tmplab = Ylab
        Y = X
        Ylab = Xlab
        X = tmp
        Xlab = tmplab
    if not ( Xlab == 'rho' and Ylab in ['T','U','S']):
        raise NotImplementedError('Calculation of', Qlab, 'from', Xlab, 'and', Ylab, 'is not available' )
    
    if Qlab == 'P':
        uconversion_Q = uconversion_P_inv
    elif Qlab == 'T':
        uconversion_Q = 1.
    elif Qlab == 'U':
        uconversion_Q = uconversion_U_inv
    elif Qlab == 'S':
        uconversion_Q = uconversion_S_inv
    elif Qlab == 'cs':
        uconversion_Q = 1.
    else:
        raise NotImplementedError('Error: calculation of', Qlab, 'not supported.')
        return None

    if Ylab == 'S':
        Y = Y*uconversion_S
    #elif Ylab == 'rho':
    #    Y = Y
    elif Ylab == 'P':
        Y = Y*uconversion_P
    elif Ylab == 'U':
        Y = Y*uconversion_U

    for i in numba.prange(len(X)):
        EOS = select(mats[i])

        if EOS.TYPE in ['ANEOS','SESAME']:
            if Ylab == 'S':
                Q[i] = tabinterp.from_rhoS(Qlab, X[i], Y[i], EOS)
            elif Ylab == 'U':
                Q[i] = tabinterp.from_rhoU(Qlab, X[i], Y[i], EOS)
            elif Ylab == 'T':
                Q[i] = tabinterp.from_rhoT(Qlab, X[i], Y[i], EOS)
        else:
            raise NotImplementedError('Non-ANEOS EOS not currently supported.')            
    return Q*uconversion_Q


