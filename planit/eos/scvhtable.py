"""
   planit SCVH table wrapper functions
"""

from ..main import *
from .eostab_extension import *
from .get_eos_data import get_eosdata_path


def loadSCVHEOS(eos='SCVH_H_He_extended', eostype='SCVH', debug = False):
    """
    READ IN SCVH EOS and fill the extEOStable class object
    
    Returns: EOStable
    """
        
    if eos == 'SCVH_H_He_extended':
        eosdir = get_eosdata_path("SCVH")
        file = 'scvh_extended_dt_hydrogen_722_helium_278.data'
        womaID = None
    #elif eos == '???':
    #    eosdir = get_eosdata_path("???")
    #    file = ''
    #    womaID = None
        
    NewEOS  = EOStable() # FIRST make new empty EOS object
    NewEOS.TYPE = eostype
    NewEOS.womaID = womaID
    NewEOS.VERSION = 1.0
    
    NewEOS.loadSCVHtable(eosdir+file)
    
    NewEOS.MODELNAME = eos

    return NewEOS

