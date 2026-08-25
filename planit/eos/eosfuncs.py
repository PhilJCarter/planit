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


def _select_user_eos(
        slot, eosname=None, eosdir=None, *, gadget_file=None,
        gadget_low_density_log=False, gadget_out_of_domain='error'):
    """Load or return a custom EOS slot without hiding invalid requests.

    A user slot can contain either a SESAME-style table directory or one
    Gadget-style density--entropy table.  A replacement is assigned to the
    module cache only after it has loaded successfully.
    """
    cache_name = _USER_EOS_CACHE_NAMES[slot]
    cached_eos = globals()[cache_name]
    has_eosname = eosname is not None
    has_eosdir = eosdir is not None
    has_gadget_file = gadget_file is not None

    if has_eosdir and has_gadget_file:
        raise ValueError(
            f'Custom EOS {slot} accepts either eosdir or gadget_file, not both.'
        )

    has_source = has_eosdir or has_gadget_file
    if has_eosname != has_source:
        raise ValueError(
            f'Custom EOS {slot} requires both eosname and eosdir for SESAME, '
            'or both eosname and gadget_file for Gadget; provide none of '
            'these arguments only to retrieve an already loaded table.'
        )

    if not has_source:
        if gadget_low_density_log or gadget_out_of_domain != 'error':
            raise ValueError(
                'Gadget interpolation options can only be set while loading '
                'a gadget_file.'
            )
        if cached_eos is None:
            raise ValueError(
                f'Custom EOS {slot} is not loaded. Provide eosname with either '
                'eosdir or gadget_file.'
            )
        return cached_eos

    # Assign only after the loader succeeds, so a malformed replacement cannot
    # leave a slot uninitialised or make select() silently return stale data.
    if has_eosdir:
        if gadget_low_density_log or gadget_out_of_domain != 'error':
            raise ValueError(
                'Gadget interpolation options cannot be used with eosdir.'
            )
        new_eos = loadANEOSEOS(
            eos=eosname,
            eostype='SESAME',
            eosdir=eosdir,
            user=True,
            womaID=USER_EOS_SLOTS[slot],
        )
    else:
        if not isinstance(eosname, str) or not eosname.strip():
            raise ValueError('Custom EOS eosname must be a non-empty string.')
        if not isinstance(gadget_low_density_log, (bool, npy.bool_)):
            raise ValueError('gadget_low_density_log must be True or False.')
        if gadget_out_of_domain not in ('error', 'clip'):
            raise ValueError(
                "gadget_out_of_domain must be either 'error' or 'clip'."
            )

        new_eos = GADtable()
        new_eos.loadstdgadget(gadget_file)
        new_eos.MODELNAME = eosname
        new_eos.womaID = USER_EOS_SLOTS[slot]
        new_eos.gadget_low_density_log = bool(gadget_low_density_log)
        new_eos.gadget_out_of_domain = gadget_out_of_domain

    globals()[cache_name] = new_eos
    return new_eos


def select(
        name, eosname=None, eosdir=None, *, gadget_file=None,
        gadget_low_density_log=False, gadget_out_of_domain='error'):
    """Return an EOS table, loading it if necessary.

    Bundled EOS tables can be selected by their established names or material
    IDs.  Custom tables use one of the five ``User0``--``User4`` slots (or
    their IDs 900--904).  A SESAME table is loaded with ``eosname`` and
    ``eosdir``::

        table = select('User0', eosname='MyMaterial', eosdir='/path/to/table')

    A Gadget density--entropy table is loaded with ``eosname`` and
    ``gadget_file``::

        table = select(
            'User0',
            eosname='MyMaterial',
            gadget_file='/path/to/Gadget_EOS.txt',
        )

    ``eosdir`` may be a string or path-like object and does not need a trailing
    slash.  Repeating this call with both arguments replaces that slot after a
    successful load.  Calling ``select('User0')`` later returns the cached
    table; calling it before a successful load, or supplying only one of the
    two arguments, raises ``ValueError``.
    """
    user_slot = _user_slot_name(name)
    if user_slot is not None:
        return _select_user_eos(
            user_slot,
            eosname=eosname,
            eosdir=eosdir,
            gadget_file=gadget_file,
            gadget_low_density_log=gadget_low_density_log,
            gadget_out_of_domain=gadget_out_of_domain,
        )
    if isinstance(name, str) and name.startswith('User'):
        raise ValueError(
            f'Unknown user EOS slot {name!r}. Supported slots are User0 through User4 '
            '(WoMa IDs 900 through 904).'
        )
    if (eosname is not None or eosdir is not None or gadget_file is not None
            or gadget_low_density_log or gadget_out_of_domain != 'error'):
        raise ValueError(
            'Custom EOS loading arguments can only be used with User0 '
            'through User4 (WoMa IDs 900 through 904).'
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
        #return None
        
        
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
       Y - array of 2nd known property values
       mats - array of material identifiers
       
       returns array of interpolated property at X, Y points
    """
    # Work on floating-point copies because unit conversion and explicit
    # Gadget endpoint clipping must not mutate arrays supplied by the caller.
    X = npy.asarray(X, dtype=float)
    Y = npy.asarray(Y, dtype=float)
    mats = npy.asarray(mats)
    if X.ndim == 0:
        X = X.reshape(1)
    if Y.ndim == 0:
        Y = Y.reshape(1)
    if mats.ndim == 0:
        mats = mats.reshape(1)

    if X.ndim != 1 or Y.ndim != 1 or mats.ndim != 1:
        raise ValueError('X, Y, and mats arrays must be one-dimensional')
    if not X.shape == Y.shape == mats.shape:
        raise ValueError('X, Y, and mats arrays must be the same size/shape')

    X = X.copy()
    Y = Y.copy()
        
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

    if X.size == 0:
        return npy.empty(0, dtype=float)

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
        material_mask = mats == mat

        if EOS.TYPE == 'GADGET':
            if Ylab != 'S' or Qlab not in ('P', 'T', 'U', 'cs'):
                raise NotImplementedError(
                    f'Gadget tables support P, T, U, or cs from rho and S; '
                    f'calculation of {Qlab} from {Xlab} and {Ylab} is not available.'
                )

            entropy_axis = npy.asarray(EOS.S)
            if entropy_axis.ndim == 2:
                entropy_axis = entropy_axis[:, 0]

            material_rho = X[material_mask]
            material_entropy = Y[material_mask]
            non_finite = ~(npy.isfinite(material_rho)
                           & npy.isfinite(material_entropy))
            rho_below = material_rho < EOS.rho[0]
            rho_above = material_rho > EOS.rho[-1]
            entropy_below = material_entropy < entropy_axis[0]
            entropy_above = material_entropy > entropy_axis[-1]
            outside = (non_finite | rho_below | rho_above
                       | entropy_below | entropy_above)

            if npy.any(non_finite):
                raise ValueError(
                    f'Gadget EOS {EOS.MODELNAME!r} received '
                    f'{npy.count_nonzero(non_finite)} non-finite rho/S query '
                    'point(s).'
                )

            if npy.any(outside):
                if EOS.gadget_out_of_domain == 'error':
                    raise ValueError(
                        f'Gadget EOS {EOS.MODELNAME!r} query is outside the '
                        f'table domain: rho below={npy.count_nonzero(rho_below)}, '
                        f'rho above={npy.count_nonzero(rho_above)}, '
                        f'S below={npy.count_nonzero(entropy_below)}, '
                        f'S above={npy.count_nonzero(entropy_above)}. '
                        f'Valid ranges are rho=[{EOS.rho[0]}, {EOS.rho[-1]}] '
                        f'g cm^-3 and S=['
                        f'{entropy_axis[0]*uconversion_S_inv}, '
                        f'{entropy_axis[-1]*uconversion_S_inv}] '
                        'erg g^-1 K^-1. Reload the slot '
                        "with gadget_out_of_domain='clip' to reproduce Gadget "
                        'endpoint clipping.'
                    )
                if EOS.gadget_out_of_domain != 'clip':
                    raise ValueError(
                        f'Unknown Gadget out-of-domain policy '
                        f'{EOS.gadget_out_of_domain!r}.'
                    )
                X[material_mask] = npy.clip(
                    material_rho, EOS.rho[0], EOS.rho[-1]
                )
                Y[material_mask] = npy.clip(
                    material_entropy, entropy_axis[0], entropy_axis[-1]
                )

            # Raising from inside Numba's parallel interpolation loop can
            # surface as a SystemError.  Check logarithmic cells here so the
            # public function gives a clear and deterministic ValueError.
            if EOS.gadget_low_density_log:
                evaluation_rho = X[material_mask]
                evaluation_entropy = Y[material_mask]
                logarithmic = evaluation_rho <= 2.0
                if npy.any(logarithmic):
                    rho_log = evaluation_rho[logarithmic]
                    entropy_log = evaluation_entropy[logarithmic]
                    rho_index = npy.clip(
                        npy.searchsorted(EOS.rho, rho_log) - 1,
                        0,
                        EOS.ND - 2,
                    )
                    entropy_index = npy.clip(
                        npy.searchsorted(entropy_axis, entropy_log) - 1,
                        0,
                        EOS.NS - 2,
                    )
                    if Qlab == 'P':
                        property_array = EOS.P
                    elif Qlab == 'T':
                        property_array = EOS.T
                    elif Qlab == 'U':
                        property_array = EOS.U
                    else:
                        property_array = EOS.cs

                    invalid_log = (
                        (entropy_log <= 0.0)
                        | (entropy_axis[entropy_index] <= 0.0)
                        | (entropy_axis[entropy_index + 1] <= 0.0)
                        | (property_array[entropy_index, rho_index] <= 0.0)
                        | (property_array[entropy_index, rho_index + 1] <= 0.0)
                        | (property_array[entropy_index + 1, rho_index] <= 0.0)
                        | (property_array[
                            entropy_index + 1, rho_index + 1
                        ] <= 0.0)
                    )
                    if npy.any(invalid_log):
                        raise ValueError(
                            'Logarithmic GADGET interpolation requires '
                            'positive entropy and property values throughout '
                            f'the selected table cell; {Qlab} has '
                            f'{npy.count_nonzero(invalid_log)} invalid query '
                            'point(s).'
                        )

        passer = EOS.make_passer_class()
        EOSlist = npy.where(material_mask,passer,EOSlist)
    
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
        elif EOSlist[i].TYPE == 'GADGET':
            if Ylab == 'S':
                Q[i] = tabinterp.from_gadget_rhoS(
                    Qlab, X[i], Y[i], EOSlist[i]
                )
    return Q
