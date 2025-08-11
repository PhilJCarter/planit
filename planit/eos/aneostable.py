from ..main import *
from .eos_table import *
from .aquatable import EOStable
import numpy as npy


def loadANEOSEOS(eos='Iron-ANEOS-SLVTv0.2G1', eostype='ANEOS', debug = False):
    """
    READ IN NEW ANEOS MODEL and fill the extEOStable class object
    
    Returns: EOStable
    """
        
    if eos == 'Iron-ANEOS-SLVTv0.2G1':
        eosdir = eospath + 'aneos-iron-2020-master/'
#
#        MODELNAME = 'Iron-ANEOS-SLVTv0.2G1'
#        # Header information must all be compatible with float format
#        MATID = 1.0        # MATID number
#        DATE = 191105.     # Date as a single 6-digit number YYMMDD
#        VERSION = 0.2      # ANEOS Parameters Version number
#        FMN = 26.          # Formula weight in atomic numbers for Fe
#        FMW = 55.847       # Formula molecular weight (g/cm3) for Fe
#        # The following define the default initial state for material in the 201 table
#        R0REF   = 8.06     # g/cm3 *** R0REF is inserted into the density array; using gamma-iron for rho0
#        K0REF   = 1.51E12  # dynes/cm2; using gamma-iron for rho0
#        T0REF   = 298.     # K -- *** T0REF is inserted into the temperature array
#        P0REF   = 1.E6     # dynes/cm2 -- this defines the principal Hugoniot calculated below

    elif eos == 'Fe85Si15-ANEOS-SLVTv0.2G1':
        eosdir = eospath + 'aneos-Fe85Si15-2020-master/'
#        # ====>>>>>> YOU NEED TO MAKE SURE THESE VALUES MATCH ANEOS.INPUT  <<<<=====
#        MODELNAME = 'Fe85Si15-ANEOS-SLVTv0.2G1'
#        # Header information must all be compatible with float format
#        MATID = 1.0        # MATID number
#        DATE = 191105.     # Date as a single 6-digit number YYMMDD
#        VERSION = 0.2      # ANEOS Parameters Version number
#        FMN = 24.20        # Formula weight in atomic numbers for Fe85Si15
#        FMW = 51.68        # Formula molecular weight (g/cm3) for Fe85Si15
#        # The following define the default initial state for material in the 201 table
#        R0REF   = 7.51     # g/cm3 *** R0REF is inserted into the density array; using gamma-iron for rho0
#        K0REF   = 1.51E12  # dynes/cm2; using gamma-iron for rho0
#        T0REF   = 298.     # K -- *** T0REF is inserted into the temperature array
#        P0REF   = 1.E6     # dynes/cm2 -- this defines the principal Hugoniot calculated below
# #       #-------------------------------------------------------------

    elif eos == 'Forsterite-ANEOS-SLVTv1.0G1':
        eosdir = eospath + 'aneos-forsterite-2019-master/'
#        MODELNAME = 'Forsterite-ANEOS-SLVTv1.0G1'
#        # Header information must all be compatible with float format
#        MATID = 1.0        # MATID number
#        DATE = 190802.     # Date as a single 6-digit number YYMMDD
#        VERSION = 0.1      # ANEOS Parameters Version number
#        FMN = 70.          # Formula weight in atomic numbers for Mg2SiO4
#        FMW = 140.691      # Formula molecular weight (g/cm3) for Mg2SiO4
#        # The following define the default initial state for material in the 201 table
#        R0REF   = 3.22     # g/cm3 *** R0REF is inserted into the density array
#        K0REF   = 1.1E12   # dynes/cm2
#        T0REF   = 298.     # K -- *** T0REF is inserted into the temperature array
#        P0REF   = 1.E6     # dynes/cm2 -- this defines the principal Hugoniot calculated below

    elif eos == '5PhaseEOSv8.3':
        eosdir = eospath + '5-phase-water/'
        MODELNAME='5PhaseEOSv8.3'
        # The following define the default initial state for material in the 201 table
        R0REF   = 1.0      # g/cm3 *** R0REF is inserted into the density array
        K0REF   = 2E10     # dynes/cm2
        T0REF   = 298.     # K -- *** T0REF is inserted into the temperature array
        P0REF   = 1.E6
    
    NewEOS  = EOStable() # FIRST make new empty EOS object
    NewEOS.TYPE = eostype
    
    if eostype=='ANEOS':
        # READ EOS HEADER INFORMATION 
        # READ IN ANEOS INPUT FILE
        aneosinputfile = open(eosdir+'ANEOS.INPUT',"r")  
        an_in=aneosinputfile.readlines()   # read in the whole ascii file at once because this is fatser
        aneosinputfile.close()
        aneoscount = 1
        for i in npy.arange(len(an_in)):
            if an_in[i].find('ANEOS') == 0:
                if aneoscount<9:
                    if debug:
                        print(' '+an_in[i-3],an_in[i-2],an_in[i-1],an_in[i])
                    aneoscount=aneoscount+1
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

    # READ SESAME HEADER
    sesfile = open(eosdir+'NEW-SESAME-STD.TXT',"r")  
    sesdata=sesfile.readlines(5000)
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
        
    NewEOS.loadextsesame(eosdir+'NEW-SESAME-EXT.TXT') # LOAD THE EXTENDED 301 SESAME FILE GENERATED BY STSM VERSION OF ANEOS
    NewEOS.loadstdsesame(eosdir+'NEW-SESAME-STD-NOTENSION.TXT') # LOAD THE STANDARD 301 SESAME FILE GENERATED BY STSM VERSION OF ANEOS
    NewEOS.MODELNAME = eos # string set above in user input
    NewEOS.MDQ = npy.zeros((NewEOS.NT,NewEOS.ND)) # makes the empty MDQ array

    # Add the header info to the table. This could be done during the loading. 
    NewEOS.MATID   = MATID
    NewEOS.DATE    = DATE
    NewEOS.VERSION = VERSION
    NewEOS.FMN     = FMN
    NewEOS.FMW     = FMW
    NewEOS.R0REF   = R0REF
    NewEOS.K0REF   = K0REF
    NewEOS.T0REF   = T0REF
    NewEOS.P0REF   = P0REF

    # Load the information from ANEOS.INPUT and ANEOS.OUTPUT
    if eostype == 'ANEOS':
        NewEOS.loadaneos(aneosinfname=eosdir+'ANEOS.INPUT',aneosoutfname=eosdir+'ANEOS.OUTPUT',silent=True)
        #
        NewEOS.calchugoniot(r0=NewEOS.R0REF,t0=NewEOS.T0REF)
        #
        # calculate the 1-bar profile; loop over temp
        NewEOS.onebar.T = npy.zeros(NewEOS.NT)
        NewEOS.onebar.S = npy.zeros(NewEOS.NT)
        NewEOS.onebar.rho = npy.zeros(NewEOS.NT)
        it0 = npy.where(NewEOS.T >= NewEOS.T0REF)[0]
        id0 = npy.arange(NewEOS.ND)#npy.where(NewEOS.rho >= 0.8*NewEOS.R0REF)[0]
        for iit in range(0,NewEOS.NT):
            NewEOS.onebar.T[iit] = NewEOS.T[iit]
            NewEOS.onebar.S[iit] = npy.interp(1.E-4,NewEOS.P[iit,id0],NewEOS.S[iit,id0])
            NewEOS.onebar.rho[iit] = npy.interp(1.E-4,NewEOS.P[iit,id0],NewEOS.rho[id0])

    return NewEOS

