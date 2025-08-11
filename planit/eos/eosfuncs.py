from ..main import *

import numpy as npy
import numba
from .eos_table import *
from .aneostable import *
from .aquatable import *
from . import tabinterp


def loadEOS(eos='Iron-ANEOS-SLVTv0.2G1', eostype='ANEOS'):
    if eostype == 'ANEOS':
        return loadANEOSEOS(eos=eos, eostype='ANEOS')
    elif eostype == 'SESAME':
        return loadANEOSEOS(eos=eos, eostype='SESAME')
    elif eostype == 'AQUA':
        return loadAQUAEOS(eos=eos, eostype='AQUA')
    else:
        print('error: unsupported EOS type:', eostype)


# Load EOS tables
ANEOSIron       = loadEOS(eos='Iron-ANEOS-SLVTv0.2G1', eostype='ANEOS')
ANEOSFeSiAlloy  = loadEOS(eos='Fe85Si15-ANEOS-SLVTv0.2G1', eostype='ANEOS')
ANEOSForsterite = loadEOS(eos='Forsterite-ANEOS-SLVTv1.0G1', eostype='ANEOS')

FivePhaseWater = loadEOS(eos='5PhaseEOSv8.3', eostype='SESAME')

#AQUAWater = loadEOS(eos='Water-AQUA-v1.0', eostype='AQUA')

ironnames  = ['iron','ANEOSIron','Fe','Iron',401]
alloynames = ['alloy','ANEOSFeSiAlloy','FeSi','Alloy','IronAlloy','ironalloy',402]
forsteritenames = ['forsterite','ANEOSForsterite','Forsterite','Fo',400]
aquawaternames = ['AQUA','AQUAWater','aqua',304]
fivephasewaternames = ['5PhaseWater','5phasewater','SS08','SenftStewartWater','SenftStewart08',303]

def select(name):
    if name in ironnames:
        return ANEOSIron
    elif name in alloynames:
        return ANEOSFeSiAlloy
    elif name in forsteritenames:
        return ANEOSForsterite
    elif name in aquawaternames:
        return AQUAWater
    elif name in fivephasewaternames:
        return FivePhaseWater
    else:
        print('Unknown EOS')
        return None
        
        
class isentrope_class:
    """Class to hold isentrope data extracted from EOS table.""" 
    def __init__(self, entropy=None, material=None): 
        """A function to initialize the class object.""" 
        self.entropy = entropy
        self.material = material
        self.ND = 0 # number of density points
        self.density     = []   
        self.pressure    = []
        self.temperature = []
        self.soundspeed  = []
        self.intenergy   = []
        self.units = ''
        if self.material and self.entropy:
            self.extract()
    
    def extract(self,material=None,entropy=None):
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


uconversion_P_inv = uconversion_P_SI2cgs
uconversion_U_inv = uconversion_U_SI2cgs
uconversion_S_inv = uconversion_S_SI2cgs
uconversion_rho = uconversion_rho_cgs2SI
uconversion_S = uconversion_S_cgs2SI
uconversion_P = uconversion_P_cgs2SI
uconversion_U = uconversion_U_cgs2SI


@numba.njit(parallel=True)
def calcprop(Qlab,Xlab,Ylab,X,Y,mats):

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
    else:
        print('error')
        return None

    if Ylab == 'S':
        Y = Y*uconversion_S
    elif Ylab == 'rho':
        Y = Y*uconversion_rho
    elif Ylab == 'P':
        Y = Y*uconversion_P
    elif Ylab == 'u':
        Y = Y*uconversion_U

    for i in numba.prange(len(X)):
        x = X[i]
        EOS = select(mats[i])

        uconversion_X = uconversion_rho
        if EOS.TYPE in ['ANEOS','SESAME']:
            if Ylab == 'S':
                Q[i] = tabinterp.from_rhoS(Qlab,x*uconversion_X, Y[i], EOS)
            elif Ylab == 'U':
                Q[i] = tabinterp.from_rhoU(Qlab,x*uconversion_X, Y[i], EOS)
            elif Ylab == 'T':
                Q[i] = tabinterp.from_rhoT(Qlab,x*uconversion_X, Y[i], EOS)
        else:
            raise NotImplementedError('Non-ANEOS EOS not currently supported')            
    return Q*uconversion_Q


