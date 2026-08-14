"""
   planit ANEOS/SESAME table wrapper functions
"""

from ..main import *
from .eos_table import *
from .eostab_extension import *
import numpy as npy
from pathlib import Path


REQUIRED_SESAME_FILES = (
    'NEW-SESAME-STD.TXT',
    'NEW-SESAME-STD-NOTENSION.TXT',
    'NEW-SESAME-EXT.TXT',
)
"""Files required for a custom planit SESAME EOS table."""

USER_WOMA_IDS = frozenset(range(900, 905))
"""WoMa IDs reserved by planit for custom EOS tables."""


def _validate_user_sesame_request(eos, eosdir, womaID):
    """Validate a custom SESAME request and return its table directory.

    The public custom-EOS interface is :func:`planit.eos.select`; this helper
    also keeps direct ``loadANEOSEOS(..., user=True)`` calls from failing later
    with an obscure file or local-variable error.
    """
    if not isinstance(eos, str) or not eos.strip():
        raise ValueError('Custom EOS eosname must be a non-empty string.')

    if (isinstance(womaID, bool) or not isinstance(womaID, int)
            or womaID not in USER_WOMA_IDS):
        raise ValueError(
            'Custom EOS womaID must be one of 900, 901, 902, 903, or 904.'
        )

    if eosdir is None:
        raise ValueError('Custom EOS eosdir must name a directory containing SESAME files.')
    if isinstance(eosdir, str) and not eosdir.strip():
        raise ValueError('Custom EOS eosdir must not be empty.')

    try:
        table_dir = Path(eosdir).expanduser()
    except TypeError as exc:
        raise ValueError('Custom EOS eosdir must be a path-like value.') from exc

    if not table_dir.exists():
        raise FileNotFoundError(
            f'Custom EOS directory does not exist: {table_dir}'
        )
    if not table_dir.is_dir():
        raise NotADirectoryError(
            f'Custom EOS path is not a directory: {table_dir}'
        )

    missing_files = [
        filename for filename in REQUIRED_SESAME_FILES
        if not (table_dir / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f'Custom EOS directory {table_dir} is missing required file(s): '
            + ', '.join(missing_files)
        )

    return table_dir


def loadANEOSEOS(eos='Iron-ANEOS-SLVTv0.2G1', eostype='ANEOS', debug=False,
                 eosdir=None, user=False, womaID=None):
    """
    Load a bundled ANEOS/SESAME table or a validated custom SESAME table.

    Custom tables are normally loaded through ``planit.eos.select``.  Direct
    callers must use ``user=True``, ``eostype='SESAME'``, a directory that
    contains :data:`REQUIRED_SESAME_FILES`, and a ``womaID`` from 900 to 904.
    ``womaID`` is optional for bundled EOS models and is ignored for them.

    Returns: EOStable
    """

    NewEOS = EOStable()  # make new empty EOS object

    if user:
        if eostype != 'SESAME':
            raise ValueError("Custom EOS tables require eostype='SESAME'.")
        eosdir = _validate_user_sesame_request(eos, eosdir, womaID)

    elif eos == 'Iron-ANEOS-SLVTv0.2G1':
        eosdir = eospath + 'aneos-iron-2020-master/'
        womaID = 401
        # MODELNAME = 'Iron-ANEOS-SLVTv0.2G1'
        # MATID = 1.0        # MATID number
        # DATE = 191105.     # Date as a single 6-digit number YYMMDD
        # VERSION = 0.2      # ANEOS Parameters Version number

    elif eos == 'Fe85Si15-ANEOS-SLVTv0.2G1':
        eosdir = eospath + 'aneos-Fe85Si15-2020-master/'
        womaID = 402
        # MODELNAME = 'Fe85Si15-ANEOS-SLVTv0.2G1'
        # MATID = 1.0        # MATID number
        # DATE = 191105.     # Date as a single 6-digit number YYMMDD
        # VERSION = 0.2      # ANEOS Parameters Version number

    elif eos == 'Forsterite-ANEOS-SLVTv1.0G1':
        eosdir = eospath + 'aneos-forsterite-2019-master/'
        womaID = 400
        # MODELNAME = 'Forsterite-ANEOS-SLVTv1.0G1'
        # MATID = 1.0        # MATID number
        # DATE = 190802.     # Date as a single 6-digit number YYMMDD
        # VERSION = 0.1      # ANEOS Parameters Version number

    elif eos == 'Pyrolite_ANEOS_SLVTv0.2':
        eosdir = eospath + 'aneos-pyrolite-2022/'
        womaID = 403
        # MODELNAME = 'Pyrolite_ANEOS_SLVTv0.2'
        # MATID   = 1.0      # MATID number
        # DATE    = 210627.  # Date as a single 6-digit number YYMMDD
        # VERSION = 0.1      # ANEOS Parameters Version number

    elif eos == '5PhaseEOSv8.3':
        eosdir = eospath + '5-phase-water/'
        womaID = 303
        # MODELNAME='5PhaseEOSv8.3'
        # The following define the default initial state for material in the 201 table
        NewEOS.R0REF = 1.0      # g/cm3 *** R0REF is inserted into the density array
        NewEOS.K0REF = 2E10     # dynes/cm2
        NewEOS.T0REF = 298.     # K -- *** T0REF is inserted into the temperature array
        NewEOS.P0REF = 1.E6

    else:
        raise ValueError('EOS:', eos, 'unknown.')

    NewEOS.TYPE = eostype
    NewEOS.womaID = womaID

    # pathlib joins work with both bundled paths and user paths, so callers do
    # not have to supply a trailing path separator for a custom EOS directory.
    eosdir = Path(eosdir)

    if eostype == 'ANEOS':
        # READ EOS HEADER INFORMATION
        # READ IN ANEOS INPUT FILE
        aneosinputfile = open(eosdir / 'ANEOS.INPUT', "r")
        an_in = aneosinputfile.readlines()   # read in the whole ascii file at once because this is fatser
        aneosinputfile.close()
        aneoscount = 1
        for i in npy.arange(len(an_in)):
            if an_in[i].find('ANEOS') == 0:
                if aneoscount < 9:
                    if debug:
                        print(' '+an_in[i-3], an_in[i-2], an_in[i-1], an_in[i])
                    aneoscount = aneoscount+1
                else:
                    if debug:
                        print(' '+an_in[i])
                if an_in[i].find('ANEOS2') == 0:
                    tmp = an_in[i]
                    R0REF = float(tmp[30:40])
                    T0REF = float(tmp[40:50])
                    P0REF = float(tmp[50:60])
                    K0REF = float(tmp[60:70])
                    break
        NewEOS.R0REF = R0REF
        NewEOS.K0REF = K0REF
        NewEOS.T0REF = T0REF
        NewEOS.P0REF = P0REF

    # READ SESAME HEADER
    sesfile = open(eosdir / 'NEW-SESAME-STD.TXT', "r")
    sesdata = sesfile.readlines(5000)
    sesfile.close()
    for i in range(len(sesdata)):
        if sesdata[i].find(' INDEX') == 0:
            tmp = sesdata[i+1]
            MATID = float(tmp.split()[0])
            DATE = float(tmp.split()[1])
            VERSION = float(tmp.split()[3])
        if sesdata[i].find(' RECORD     TYPE =  201') == 0:
            tmp = sesdata[i+1]
            FMN = float(tmp.split()[0])
            FMW = float(tmp.split()[1])
            break

    NewEOS.loadextsesame(str(eosdir / 'NEW-SESAME-EXT.TXT'))  # LOAD THE EXTENDED 301 SESAME FILE GENERATED BY STSM VERSION OF ANEOS
    NewEOS.loadstdsesame(str(eosdir / 'NEW-SESAME-STD-NOTENSION.TXT'))  # LOAD THE STANDARD 301 SESAME FILE GENERATED BY STSM VERSION OF ANEOS
    NewEOS.MODELNAME = eos  # string set above in user input
    NewEOS.MDQ = npy.zeros((NewEOS.NT,NewEOS.ND))  # makes the empty MDQ array

    # Add the header info to the table. This could be done during the loading. 
    NewEOS.MATID   = MATID
    NewEOS.DATE    = DATE
    NewEOS.VERSION = VERSION
    NewEOS.FMN     = FMN
    NewEOS.FMW     = FMW

    # Load the information from ANEOS.INPUT and ANEOS.OUTPUT
    if eostype == 'ANEOS':
        NewEOS.loadaneos(aneosinfname=str(eosdir / 'ANEOS.INPUT'), aneosoutfname=str(eosdir / 'ANEOS.OUTPUT'), silent=True)
        #
        NewEOS.calchugoniot(r0=NewEOS.R0REF,t0=NewEOS.T0REF)
        #
        # calculate the 1-bar profile; loop over temp
        NewEOS.onebar.T = npy.zeros(NewEOS.NT)
        NewEOS.onebar.S = npy.zeros(NewEOS.NT)
        NewEOS.onebar.rho = npy.zeros(NewEOS.NT)
        #it0 = npy.where(NewEOS.T >= NewEOS.T0REF)[0]
        id0 = npy.arange(NewEOS.ND)#npy.where(NewEOS.rho >= 0.8*NewEOS.R0REF)[0]
        for iit in range(0,NewEOS.NT):
            NewEOS.onebar.T[iit] = NewEOS.T[iit]
            NewEOS.onebar.S[iit] = npy.interp(1.E-4, NewEOS.P[iit,id0], NewEOS.S[iit,id0])
            NewEOS.onebar.rho[iit] = npy.interp(1.E-4, NewEOS.P[iit,id0], NewEOS.rho[id0])

    return NewEOS
