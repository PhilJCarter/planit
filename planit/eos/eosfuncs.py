"""
   planit equation of state functions
"""

from ..main import *
from .eos_table import *
from .eos_table import isentrope_class as eos_isentrope_class
from .eostab_extension import *
from .aneostable import *
from .aquatable import *
from .rhoUtable import *
from . import tabinterp

import numpy as npy
import numba
from scipy import interpolate


def loadEOS(eos='Iron-ANEOS-SLVTv0.2G1', eostype='ANEOS'):
    """
       Wrapper function for loading EoS
    """
    if eostype == 'ANEOS':
        return loadANEOSEOS(eos=eos, eostype='ANEOS')
    elif eostype == 'SESAME':
        return loadANEOSEOS(eos=eos, eostype='SESAME')
    elif eostype == 'AQUA':
        return loadAQUAEOS(eos=eos, eostype='AQUA')
    elif eostype == 'HM80':
        return loadrhoUEOS(eos=eos, eostype='HM80')
    else:
        raise ValueError('Error: unsupported EOS type:', eostype)


# Variables to hold EOS tables
ANEOSIron       = None
ANEOSFeSiAlloy  = None
ANEOSForsterite = None
ANEOSPyrolite   = None

FivePhaseWater  = None
AQUAWater = None
HM80HHe   = None

UserEOS0 = None
UserEOS1 = None
UserEOS2 = None
UserEOS3 = None
UserEOS4 = None


# Name lists for EoS
ironnames  = ['Iron-ANEOS-SLVTv0.2G1', 'iron', 'ANEOSIron', 'Fe', 'Iron', 401]
alloynames = ['Fe85Si15-ANEOS-SLVTv0.2G1', 'alloy', 'ANEOSFeSiAlloy', 'FeSi', 'Alloy', 'IronAlloy', 'ironalloy', 402]
forsteritenames = ['Forsterite-ANEOS-SLVTv1.0G1', 'forsterite', 'ANEOSForsterite', 'Forsterite', 'Fo', 400]
pyrolitenames = ['Pyrolite_ANEOS_SLVTv0.2', 'pyrolite', 'Pyrolite', 'ANEOSPyrolite', 403]
aquawaternames = ['Water-AQUA-v1.0', 'AQUA', 'AQUAWater', 'aqua', 304]
fivephasewaternames = ['5PhaseEOSv8.3', '5PhaseWater', '5phasewater', 'SS08', 'SenftStewartWater', 'SenftStewart08', 303]
hm80HHenames = ['HM80-HHe-v2.0', 'HM80_HHe', 'HM80HHe', 200]

user0names = ['User0', 900]
user1names = ['User1', 901]
user2names = ['User2', 902]
user3names = ['User3', 903]
user4names = ['User4',904]

USER_EOS_SLOTS = {
    'User0': 900,
    'User1': 901,
    'User2': 902,
    'User3': 903,
    'User4': 904,
}
"""Custom EOS slot names and their reserved SWIFT/WoMa material IDs."""

_USER_EOS_CACHE_NAMES = {
    slot: f'UserEOS{index}'
    for index, slot in enumerate(USER_EOS_SLOTS)
}


def _user_slot_name(name):
    """Return the canonical user-slot name for a name or WoMa ID."""
    for slot, womaID in USER_EOS_SLOTS.items():
        if name == slot or name == womaID:
            return slot
    return None


def _select_user_eos(slot, eosname=None, eosdir=None):
    """Load or return a custom EOS slot without hiding invalid requests."""
    cache_name = _USER_EOS_CACHE_NAMES[slot]
    cached_eos = globals()[cache_name]
    has_eosname = eosname is not None
    has_eosdir = eosdir is not None

    if has_eosname != has_eosdir:
        raise ValueError(
            f'Custom EOS {slot} requires both eosname and eosdir; '
            'provide neither only to retrieve an already loaded table.'
        )

    if not has_eosname:
        if cached_eos is None:
            raise ValueError(
                f'Custom EOS {slot} is not loaded. Provide both eosname and eosdir.'
            )
        return cached_eos

    # Assign only after the loader succeeds, so a malformed replacement cannot
    # leave a slot uninitialised or make select() silently return stale data.
    new_eos = loadANEOSEOS(
        eos=eosname,
        eostype='SESAME',
        eosdir=eosdir,
        user=True,
        womaID=USER_EOS_SLOTS[slot],
    )
    globals()[cache_name] = new_eos
    return new_eos


def select(name, eosname=None, eosdir=None):
    """Return an EOS table, loading it if necessary.

    Bundled EOS tables can be selected by their established names or material
    IDs.  Custom SESAME tables use one of the five ``User0``--``User4`` slots
    (or their IDs 900--904) and must be loaded with both ``eosname`` and
    ``eosdir``::

        table = select('User0', eosname='MyMaterial', eosdir='/path/to/table')

    ``eosdir`` may be a string or path-like object and does not need a trailing
    slash.  Repeating this call with both arguments replaces that slot after a
    successful load.  Calling ``select('User0')`` later returns the cached
    table; calling it before a successful load, or supplying only one of the
    two arguments, raises ``ValueError``.
    """
    user_slot = _user_slot_name(name)
    if user_slot is not None:
        return _select_user_eos(user_slot, eosname=eosname, eosdir=eosdir)
    if isinstance(name, str) and name.startswith('User'):
        raise ValueError(
            f'Unknown user EOS slot {name!r}. Supported slots are User0 through User4 '
            '(WoMa IDs 900 through 904).'
        )

    if name in ironnames:
        global ANEOSIron
        if not ANEOSIron:
            ANEOSIron = loadEOS(eos='Iron-ANEOS-SLVTv0.2G1', eostype='ANEOS')
        return ANEOSIron
    elif name in alloynames:
        global ANEOSFeSiAlloy
        if not ANEOSFeSiAlloy:
            ANEOSFeSiAlloy = loadEOS(eos='Fe85Si15-ANEOS-SLVTv0.2G1', eostype='ANEOS')
        return ANEOSFeSiAlloy
    elif name in forsteritenames:
        global ANEOSForsterite
        if not ANEOSForsterite:
            ANEOSForsterite = loadEOS(eos='Forsterite-ANEOS-SLVTv1.0G1', eostype='ANEOS')
        return ANEOSForsterite
    elif name in pyrolitenames:
        global ANEOSPyrolite
        if not ANEOSPyrolite:
            ANEOSPyrolite = loadEOS(eos='Pyrolite_ANEOS_SLVTv0.2', eostype='ANEOS')
        return ANEOSPyrolite
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
    else:
        raise ValueError('Unknown EOS:', name)
        
        
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
        EOS = select(self.material)
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

    EOSlist = npy.empty(len(X),dtype=object)
    for mat in npy.unique(mats):
        EOS = select(mat)
        passer = EOS.make_passer_class()
        EOSlist = npy.where(mats==mat,passer,EOSlist)
    
    Q = _calc_prop(Qlab,Xlab,Ylab,X,Y,EOSlist.tolist())
    return Q*uconversion_Q


@numba.njit(parallel=True)
def _calc_prop(Qlab,Xlab,Ylab,X,Y,EOSlist):
    #print(Qlab,Xlab,Ylab)
    Q = npy.zeros(len(X))
    for i in numba.prange(len(X)):

        if EOSlist[i].TYPE in ['ANEOS','SESAME','AQUA']:
            if Ylab == 'S':
                Q[i] = tabinterp.from_rhoS(Qlab, X[i], Y[i], EOSlist[i])
            elif Ylab == 'U':
                Q[i] = tabinterp.from_rhoU(Qlab, X[i], Y[i], EOSlist[i])
            elif Ylab == 'T':
                Q[i] = tabinterp.from_rhoT(Qlab, X[i], Y[i], EOSlist[i])
        elif EOSlist[i].TYPE == 'HM80':
            if (Ylab != 'U') or (Qlab not in ['P','T']):
                #raise NotImplementedError('Calculation of', Qlab, 'from', Xlab, 'and', Ylab, 'is not available' )
                #print('Calculation of', Qlab, 'from', Xlab, 'and', Ylab, 'is not available. Returning NaN.')
                Q[i] = npy.nan
            else:
                Q[i] = tabinterp.from_rhoU1D(Qlab, X[i], Y[i], EOSlist[i])
    return Q
