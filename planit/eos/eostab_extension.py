"""
   planit eos_table.extEOStable class extensions
"""

from .eos_table import *
import numpy as npy
import numba

# numba type def for EOS passer class
EOSpasser_spec = [
    ('ND', numba.types.int64),
    ('NT', numba.types.int64),
    ('rho', numba.types.float64[:]),
    ('T', numba.types.float64[:]),
    ('P', numba.types.float64[:, :]),
    ('U', numba.types.float64[:, :]),
    ('A', numba.types.float64[:, :]),
    ('S', numba.types.float64[:, :]),
    ('cs', numba.types.float64[:, :]),
    ('cv', numba.types.float64[:, :]),
#    ('KPA', numba.types.float64[:, :]),
#    ('MDQ', numba.types.float64[:, :]),
    ('TYPE', numba.types.unicode_type),
    ('womaID', numba.types.int64),
    ('NU', numba.types.int64),
    ('U_1D', numba.types.float64[:]),
    ('T_2D', numba.types.float64[:, :])
]


@numba.experimental.jitclass(EOSpasser_spec)
class EOSpasser():
    """
       Numba-compatible class for passing EOS data to interpolation functions
    """
    def __init__(self,ND,NT,NU):
        self.ND = ND
        self.NT = NT
        self.rho = npy.zeros(self.ND)
        self.T   = npy.zeros(self.NT)
        self.P   = npy.zeros((self.ND,self.NT))
        self.U   = npy.zeros((self.ND,self.NT))
        self.A   = npy.zeros((self.ND,self.NT))
        self.S   = npy.zeros((self.ND,self.NT))
        self.cs  = npy.zeros((self.ND,self.NT))
        self.cv  = npy.zeros((self.ND,self.NT))
        # self.KPA = np.zeros((self.ND,self.NT))
        # self.MDQ = np.zeros(self.ND*self.NT)
        self.TYPE = ''
        self.womaID = 0
        self.NU = NU
        self.U_1D = np.zeros(self.NU)
        self.T_2D = np.zeros((self.NU,self.ND))


# @numba.experimental.jitclass(extEOStable_spec+EOStable_spec)
class EOStable(extEOStable):
    """
       Adds TYPE and woma numerical ID fields, and rho-U table compatibility 
       to eos_table.extEOStable class
    """
    # __init__ext = extEOStable.__init__
    def __init__(self):
        extEOStable.__init__(self)
        # self.__init__ext
        self.TYPE = ''
        self.womaID = 0
        self.NU = 0    # only needed for rho-U format tables
        self.U_1D = npy.zeros(self.NU)  # only needed for rho-U format tables
        self.T_2D = npy.zeros((self.NU,self.ND))
        # self.name = ''
    
    def make_passer_class(self):
        """
           Construct numba-compatible passer class to pass EOS data to interpolation functions
        """
        passer = EOSpasser(self.ND,self.NT,self.NU)
        passer.rho = self.rho
        passer.T = self.T
        passer.P = self.P
        passer.U = self.U
        passer.A = self.A
        passer.S = self.S
        passer.cs = self.cs
        passer.cv = self.cv
        # passer.KPA = self.KPA
        # passer.MDQ = self.MDQ
        passer.TYPE = self.TYPE
        passer.womaID = self.womaID
        passer.U_1D = self.U_1D
        return passer
        
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

    def loadrhoUtable(self,fname):
        with open(fname,'r') as tablefile:
            ND = None
            NU = None
            for i in range(13):
                tmp = tablefile.readline()
            lrhomin,lrhomax,ND,lUmin,lUmax,NU = tmp.split()
        self.ND = int(ND)
        self.NU = int(NU)
        self.rho = npy.exp(npy.linspace(float(lrhomin),float(lrhomax),self.ND))/1000.
        self.U_1D = npy.exp(npy.linspace(float(lUmin),float(lUmax),self.NU))/1.e6
        data = npy.loadtxt(fname,skiprows=13,unpack=False)
        self.P = data[:self.ND].reshape(self.ND,self.NU).T/1.e9
        self.T_2D = data[self.ND:].reshape(self.ND,self.NU).T


# EOShugoniot_spec = [
#     ('NH', numba.types.int32),
#     ('rho0', numba.types.float64),
#     ('rho0_err', numba.types.float64),
#     ('T0', numba.types.float64),
#     ('rho', numba.types.float64[:]),
#     ('rho_err', numba.types.float64[:]),
#     ('T', numba.types.float64[:]),
#     ('T_err', numba.types.float64[:]),
#     ('P', numba.types.float64[:]),
#     ('P_err', numba.types.float64[:]),
#     ('U', numba.types.float64[:]),
#     ('S', numba.types.float64[:]),
#     ('S_err', numba.types.float64[:]),
#     ('up', numba.types.float64[:]),
#     ('up_err', numba.types.float64[:]),
#     ('us', numba.types.float64[:]),
#     ('us_err', numba.types.float64[:]),
#     ('cs', numba.types.float64[:]),
#     ('ref', numba.types.float64[:]),
#     ('ref_err', numba.types.float64[:]),
#     ('gamma', numba.types.float64[:]),
#     ('gamma_err', numba.types.float64[:]),
#     ('units', numba.types.unicode_type),
#     ('label', numba.types.unicode_type)
# ]
# 
# @numba.experimental.jitclass(EOShugoniot_spec)
# class EOShugoniot(EOShugoniot):
#     pass
# 
# EOSvc_spec = [
#     ('NT', numba.types.int32),
#     ('NV', numba.types.int32),
#     ('T', numba.types.float64[:]),
#     ('rl', numba.types.float64[:]),
#     ('rv', numba.types.float64[:]),
#     ('Pl', numba.types.float64[:]),
#     ('Pv', numba.types.float64[:]),
#     ('Ul', numba.types.float64[:]),
#     ('Uv', numba.types.float64[:]),
#     ('Sl', numba.types.float64[:]),
#     ('Sv', numba.types.float64[:]),
#     ('Gl', numba.types.float64[:]),
#     ('Gv', numba.types.float64[:]),
#     ('units', numba.types.unicode_type)
# ]
# 
# @numba.experimental.jitclass(EOSvc_spec)
# class EOSvaporcurve(EOSvaporcurve):
#     pass
# 
# EOSmc_spec = [
#     ('NT', numba.types.int32),
#     ('NV', numba.types.int32),
#     ('T', numba.types.float64[:]),
#     ('Tl', numba.types.float64[:]),
#     ('Ts', numba.types.float64[:]),
#     ('rl', numba.types.float64[:]),
#     ('rs', numba.types.float64[:]),
#     ('Pl', numba.types.float64[:]),
#     ('Ps', numba.types.float64[:]),
#     ('Ul', numba.types.float64[:]),
#     ('Us', numba.types.float64[:]),
#     ('Sl', numba.types.float64[:]),
#     ('Ss', numba.types.float64[:]),
#     ('units', numba.types.unicode_type)
# ]
# 
# @numba.experimental.jitclass(EOSmc_spec)
# class EOSmeltcurve(EOSmeltcurve):
#     pass
# 
# EOScp_spec = [
#     ('P', numba.types.float64),
#     ('S', numba.types.float64),
#     ('T', numba.types.float64),
#     ('rho', numba.types.float64),
#     ('U', numba.types.float64),
#     ('units', numba.types.unicode_type),
#     ('label', numba.types.unicode_type)
# ]
# 
# @numba.experimental.jitclass(EOScp_spec)
# class EOScriticalpoint(EOScriticalpoint):
#     pass
# 
# EOStp_spec = [
#     ('P', numba.types.float64),
#     ('T', numba.types.float64),
#     ('Sim', numba.types.float64),
#     ('Scm', numba.types.float64),
#     ('Siv', numba.types.float64),
#     ('Scv', numba.types.float64),
#     ('rhol', numba.types.float64),
#     ('units', numba.types.unicode_type),
#     ('label', numba.types.unicode_type)
# ]
# 
# @numba.experimental.jitclass(EOStp_spec)
# class EOStriplepoint(EOStriplepoint):
#     pass
# 
# EOS1bc_spec = [
#     ('NT', numba.types.int32),
#     ('S', numba.types.float64[:]),
#     ('T', numba.types.float64[:]),
#     ('Tvap', numba.types.float64),
#     ('Tmelt', numba.types.float64),
#     ('Sim', numba.types.float64),
#     ('Scm', numba.types.float64),
#     ('Siv', numba.types.float64),
#     ('Scv', numba.types.float64),
#     ('rhoiv', numba.types.float64),
#     ('rhocv', numba.types.float64),
#     ('rhocm', numba.types.float64),
#     ('rhoim', numba.types.float64),
#     ('units', numba.types.unicode_type),
#     ('label', numba.types.unicode_type)
# ]
# 
# @numba.experimental.jitclass(EOS1bc_spec)
# class EOS1barcurve(EOS1barcurve):
#     pass
# 
# EOSaneoshug_spec = [
#     ('ND', numba.types.int32),
#     ('NV', numba.types.int32),
#     ('rho', numba.types.float64[:]),
#     ('T', numba.types.float64[:]),
#     ('P', numba.types.float64[:]),
#     ('U', numba.types.float64[:]),
#     ('S', numba.types.float64[:]),
#     ('us', numba.types.float64[:]),
#     ('up', numba.types.float64[:]),
#     ('units', numba.types.unicode_type)
# ]
# 
# @numba.experimental.jitclass(EOSaneoshug_spec)
# class EOSaneoshugoniot(EOSaneoshugoniot):
#     pass
# 
# extEOStable_spec = [
#     ('ND', numba.types.int32),
#     ('NT', numba.types.int32),
#     ('rho', numba.types.float64[:]),
#     ('T', numba.types.float64[:]),
#     ('P', numba.types.float64[:, :]),
#     ('U', numba.types.float64[:, :]),
#     ('A', numba.types.float64[:, :]),
#     ('S', numba.types.float64[:, :]),
#     ('cs', numba.types.float64[:, :]),
#     ('cv', numba.types.float64[:, :]),
#     ('KPA', numba.types.float64[:, :]),
#     ('MDQ', numba.types.float64[:, :]),
#     ('units', numba.types.unicode_type),
#     ('hug', EOShugoniot.class_type.instance_type),
#     ('hugo', EOShugoniot.class_type.instance_type),
#     ('vc', EOSvaporcurve.class_type.instance_type),
#     ('mc', EOSmeltcurve.class_type.instance_type),
#     ('cp', EOScriticalpoint.class_type.instance_type),
#     ('tp', EOStriplepoint.class_type.instance_type),
#     ('onebar', EOS1barcurve.class_type.instance_type),
#     ('anhug', EOSaneoshugoniot.class_type.instance_type),
#     ('MATID', numba.types.float64),
#     ('DATE', numba.types.float64),
#     ('VERSION', numba.types.float64),
#     ('FMN', numba.types.float64),
#     ('FMW', numba.types.float64),
#     ('R0REF', numba.types.float64),
#     ('K0REF', numba.types.float64),
#     ('T0REF', numba.types.float64),
#     ('P0REF', numba.types.float64),
#     ('CS0REF', numba.types.float64),
#     ('gamma0', numba.types.float64),
#     ('theta0', numba.types.float64),
#     ('C24', numba.types.float64),
#     ('C60', numba.types.float64),
#     ('C61', numba.types.float64),
#     ('beta', numba.types.float64),
#     ('MODELNAME', numba.types.unicode_type),
# ]
# 
# ## @numba.experimental.jitclass(extEOStable_spec)
# ## class extEOStable(extEOStable):
# ##     pass
# 
# EOStable_spec = [
#     ('TYPE', numba.types.unicode_type),
#     ('womaID', numba.types.int32),
#     ('NU', numba.types.int32),
#     ('U_1D', numba.types.float64[:])
# ]
