from ..main import *

import numpy as npy
import numba
from .eos_table import *
from .eostab_extension import *
from .aneostable import *
from .aquatable import *
from .rhoUtable import *
from . import tabinterp


def loadEOS(eos='Iron-ANEOS-SLVTv0.2G1', eostype='ANEOS'):
    if eostype == 'ANEOS':
        return loadANEOSEOS(eos=eos, eostype='ANEOS')
    elif eostype == 'SESAME':
        return loadANEOSEOS(eos=eos, eostype='SESAME')
    elif eostype == 'AQUA':
        return loadAQUAEOS(eos=eos, eostype='AQUA')
    elif eostype == 'HM80':
        return loadrhoUEOS(eos=eos, eostype='HM80')
    else:
        print('error: unsupported EOS type:', eostype)


# Load EOS tables
ANEOSIron       = loadEOS(eos='Iron-ANEOS-SLVTv0.2G1', eostype='ANEOS')
ANEOSFeSiAlloy  = loadEOS(eos='Fe85Si15-ANEOS-SLVTv0.2G1', eostype='ANEOS')
ANEOSForsterite = loadEOS(eos='Forsterite-ANEOS-SLVTv1.0G1', eostype='ANEOS')

FivePhaseWater = None
AQUAWater = None
HM80HHe = None

UserEOS0 = None
UserEOS1 = None
UserEOS2 = None
UserEOS3 = None
UserEOS4 = None

ironnames  = ['iron','ANEOSIron','Fe','Iron',401]
alloynames = ['alloy','ANEOSFeSiAlloy','FeSi','Alloy','IronAlloy','ironalloy',402]
forsteritenames = ['forsterite','ANEOSForsterite','Forsterite','Fo',400]
aquawaternames = ['AQUA','AQUAWater','aqua',304]
fivephasewaternames = ['5PhaseWater','5phasewater','SS08','SenftStewartWater','SenftStewart08',303]
hm80HHenames = ['HM80_HHe','HM80HHe',200]

user0names = ['User0',900]
user1names = ['User1',901]
user2names = ['User2',902]
user3names = ['User3',903]
user4names = ['User4',904]

def select(name, eosname=None, eosdir=None):
    if name in ironnames:
        return ANEOSIron
    elif name in alloynames:
        return ANEOSFeSiAlloy
    elif name in forsteritenames:
        return ANEOSForsterite
    elif name in aquawaternames:
        global AQUAWater
        if not AQUAWater:
            AQUAWater = loadEOS(eos='Water-AQUA-v1.0', eostype='AQUA')
        return AQUAWater
    elif name in fivephasewaternames:
        global FivePhaseWater
        if not FivePhaseWater:
            FivePhaseWater = loadEOS(eos='5PhaseEOSv8.3', eostype='SESAME')
        return FivePhaseWater
    elif name in hm80HHenames:
        global HM80HHe
        if not HM80HHe:
            HM80HHe = loadEOS(eos='HM80-HHe-v2.0', eostype='HM80')
        return HM80HHe
    elif name in user0names:
        global UserEOS0
        if (not UserEOS0) or (eosdir and eosname):
            UserEOS0 = loadANEOSEOS(eos=eosname, eostype='SESAME', eosdir=eosdir, user=True)
        return UserEOS0
    elif name in user1names:
        global UserEOS1
        if (not UserEOS1) or (eosdir and eosname):
            UserEOS1 = loadANEOSEOS(eos=eosname, eostype='SESAME', eosdir=eosdir, user=True)
        return UserEOS1
    elif name in user2names:
        global UserEOS2
        if (not UserEOS2) or (eosdir and eosname):
            UserEOS2 = loadANEOSEOS(eos=eosname, eostype='SESAME', eosdir=eosdir, user=True)
        return UserEOS2
    elif name in user3names:
        global UserEOS3
        if (not UserEOS3) or (eosdir and eosname):
            UserEOS3 = loadANEOSEOS(eos=eosname, eostype='SESAME', eosdir=eosdir, user=True)
        return UserEOS3
    elif name in user4names:
        global UserEOS4
        if (not UserEOS4) or (eosdir and eosname):
            UserEOS4 = loadANEOSEOS(eos=eosname, eostype='SESAME', eosdir=eosdir, user=True)
        return UserEOS4
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


# cgs / eos_table unit conversions
uconversion_S = uconversion_S_cgs2SI/1e6
uconversion_P = uconversion_P_cgs2SI/1e9
uconversion_U = uconversion_U_cgs2SI/1e6

uconversion_P_inv = 1./uconversion_P
uconversion_U_inv = 1./uconversion_U
uconversion_S_inv = 1./uconversion_S


#@numba.njit(parallel=True)
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

        if EOS.TYPE in ['ANEOS','SESAME','AQUA']:
            if Ylab == 'S':
                Q[i] = tabinterp.from_rhoS(Qlab, X[i], Y[i], EOS)
            elif Ylab == 'U':
                Q[i] = tabinterp.from_rhoU(Qlab, X[i], Y[i], EOS)
            elif Ylab == 'T':
                Q[i] = tabinterp.from_rhoT(Qlab, X[i], Y[i], EOS)
        elif EOS.TYPE == 'HM80':
            if (Ylab is not 'U') or (Qlab not in ['P','T']):
                raise NotImplementedError('Calculation of', Qlab, 'from', Xlab, 'and', Ylab, 'is not available' )
            Q[i] = tabinterp.from_rhoU1D(Qlab, X[i], Y[i], EOS)
        else:
            raise NotImplementedError('Non-ANEOS EOS not currently supported.')            
    return Q*uconversion_Q


