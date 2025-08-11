from ..main import *
#from . import eos

import numpy as npy
import numba
from .eos_table import *
from .aneostable import *
from . import tabinterp


def loadEOS(eos='Iron-ANEOS-SLVTv0.2G1', eostype='ANEOS'):
    if eostype == 'ANEOS':
        return loadANEOSEOS(eos=eos, eostype='ANEOS')
    elif eostype == 'SESAME':
        return loadANEOSEOS(eos=eos, eostype='SESAME')
    else:
        print('error: unsupported EOS type:', eostype)


# Load EOS tables
ANEOSIron       = loadEOS(eos='Iron-ANEOS-SLVTv0.2G1', eostype='ANEOS')
ANEOSFeSiAlloy  = loadEOS(eos='Fe85Si15-ANEOS-SLVTv0.2G1', eostype='ANEOS')
ANEOSForsterite = loadEOS(eos='Forsterite-ANEOS-SLVTv1.0G1', eostype='ANEOS')


ironnames  = ['iron','ANEOSIron','Fe','Iron',401]
alloynames = ['alloy','ANEOSFeSiAlloy','FeSi','Alloy','IronAlloy','ironalloy',402]
forsteritenames = ['forsterite','ANEOSForsterite','Forsterite','Fo',400]


def select(name):
    if name in ironnames:
        return ANEOSIron
    elif name in alloynames:
        return ANEOSFeSiAlloy
    elif name in forsteritenames:
        return ANEOSForsterite
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


# class EOShugoniot:
#     """Class for Hugoniot array from extEOStable."""	
#     def __init__(self):
#         self.NH = 0
#         self.rho = npy.zeros(self.NH)   
#         self.T = npy.zeros(self.NH)   
#         self.P = npy.zeros(self.NH)   
#         self.U = npy.zeros(self.NH)   
#         self.S = npy.zeros(self.NH)   
#         self.up = npy.zeros(self.NH)   
#         self.us = npy.zeros(self.NH)
#         self.cs = npy.zeros(self.NH)
#         self.units = ''
# 
# class EOSvaporcurve:
#     """Class for vapor curve from ANEOS."""	
#     def __init__(self):
#         self.NT = 0
#         self.NV = 0
#         self.T = npy.zeros(self.NT)  
#         self.rl = npy.zeros(self.NT)  
#         self.rv = npy.zeros(self.NT)  
#         self.Pl = npy.zeros(self.NT)  
#         self.Pv = npy.zeros(self.NT)  
#         self.Ul = npy.zeros(self.NT)  
#         self.Uv = npy.zeros(self.NT)  
#         self.Sl = npy.zeros(self.NT)  
#         self.Sv = npy.zeros(self.NT)
#         self.Gl = npy.zeros(self.NT)  
#         self.Gv = npy.zeros(self.NT)
#         self.units = ''
# 
# class EOSmeltcurve:
#     """Class for melt curve from ANEOS."""	
#     def __init__(self):
#         self.NT = 0
#         self.NV = 0
#         self.T  = npy.zeros(self.NT)  
#         self.rl = npy.zeros(self.NT)  
#         self.rs = npy.zeros(self.NT)  
#         self.Pl = npy.zeros(self.NT)  
#         self.Ps = npy.zeros(self.NT)  
#         self.Ul = npy.zeros(self.NT)  
#         self.Us = npy.zeros(self.NT)  
#         self.Sl = npy.zeros(self.NT)  
#         self.Ss = npy.zeros(self.NT)
#         self.units = ''
# 
# class EOS1barcurve:
#     """Class for 1bar curve from the EOS."""	
#     def __init__(self):
#         self.NT    = 0
#         self.S     = npy.zeros(self.NT)  
#         self.T     = npy.zeros(self.NT)  
#         self.rho   = npy.zeros(self.NT)
#         self.Tvap  = 0.
#         self.Tmelt = 0.
#         self.Sim   = 0.
#         self.Scm   = 0.
#         self.Siv   = 0.
#         self.Scv   = 0.
#         self.rhoiv   = 0.
#         self.rhocv   = 0.
#         self.rhocm   = 0.
#         self.rhoim   = 0.
#         self.units = ''
# 
# class EOScriticalpoint:
#     """Class for critical point state from the EOS."""	
#     def __init__(self):
#         self.P   = 0
#         self.S   = 0  
#         self.T   = 0 
#         self.rho = 0
#         self.U   = 0
#         self.units = ''
# 
# class EOStriplepoint:
#     """Class for triple point state from the EOS."""	
#     def __init__(self):
#         self.P   = 0
#         self.T   = 0 
#         self.Sim   = 0.
#         self.Scm   = 0.
#         self.Siv   = 0.
#         self.Scv   = 0.
#         self.rhol  = 0.
#         self.units = ''
# 
# class EOSaneoshugoniot:
#     """Class for Hugoniot calculated in ANEOS."""	
#     def __init__(self):
#         self.ND  = 0
#         self.NV  = 0
#         #self.all = npy.zeros((self.ND,self.NV))
#         self.rho = 0
#         self.T   = 0
#         self.P   = 0
#         self.U   = 0
#         self.S   = 0
#         self.us  = 0
#         self.up  = 0
#         self.units = ''
# 
# 
# class extEOStable:
#     """Class for accessing EXTENDED SESAME-STYLE EOS tables output from ANEOS"""
#     #     ANEOS KPA FLAG
#     #                                TABLE          ANEOS
#     #     KPAQQ=STATE INDICATOR      =1, 1p    =1, 1p    (eos without melt)
#     #                                =2, 2p lv =2, 2p liquid/solid plus vapor
#     #                                          =4, 1p solid  (eos with melt)
#     #                                          =5, 2p melt   (eos with melt)
#     #                                          =6, 1p liquid (eos with melt)
#     #                                =-1 bad value of temperature
#     #                                =-2 bad value of density
#     #                                =-3 bad value of material number
#     #
#     def __init__(self):
#         self.ND  = 0 # number of density points in grid
#         self.NT  = 0 # number of temperature points in grid
#         self.rho = npy.zeros(self.ND)          # g/cm3, density values
#         self.T   = npy.zeros(self.NT)          # K, temperature values
#         self.P   = npy.zeros(self.ND*self.NT)  # GPA, pressure(T,rho)
#         self.U   = npy.zeros(self.ND*self.NT)  # MJ/kg, sp. internal energy(T,rho)
#         self.A   = npy.zeros(self.ND*self.NT)  # MJ/kg, Helmholtz free energy(T,rho)
#         self.S   = npy.zeros(self.ND*self.NT)  # MJ/K/kg, sp. entropy(T,rho)
#         self.cs  = npy.zeros(self.ND*self.NT)  # cm/s, sound speed(T,rho)
#         self.cv  = npy.zeros(self.ND*self.NT)  # MJ/K/kg, sp. heat capacity(T,rho)
#         self.KPA = npy.zeros(self.ND*self.NT)  # integer, ANEOS KPA flag(T,rho)
#         self.MDQ = npy.zeros(self.ND*self.NT)  # integer, Model Development Quality Flag(T,rho)
#         self.units = ''
#         self.hug = EOShugoniot()
#         self.vc  = EOSvaporcurve()
#         self.mc  = EOSmeltcurve()
#         self.cp  = EOScriticalpoint()
#         self.tp  = EOStriplepoint()
#         self.onebar = EOS1barcurve()
#         self.anhug = EOSaneoshugoniot()
#         # these are variables needed for the sesame header
#         self.MATID   = 0.
#         self.DATE    = 0.
#         self.VERSION = 0.
#         self.FMN     = 0.
#         self.FMW     = 0.
#         self.R0REF   = 0.
#         self.K0REF   = 0.
#         self.T0REF   = 0.
#         self.P0REF   = 0.
#         # variables needed for the ANEOS gamma function
#         self.gamma0 = 0.
#         self.theta0 = 0.
#         self.C24    = 0.
#         self.C60    = 0.
#         self.C61    = 0.
#         self.beta   = 0.
#         # model name/version string
#         self.MODELNAME = ''
# 
#     def loadstdsesame(self, fname, unitstxt=None):
#         """Function for loading STD SESAME-STYLE EOS table output from ANEOS"""
#         data = ([])
#         if unitstxt is None:
#             self.units = 'Units: rho g/cm3, T K, P GPa, U MJ/kg, A MJ/kg, S MJ/K/kg, cs cm/s, cv MJ/K/kg, KPA flag. 2D arrays are (NT,ND).'
#         else:
#             self.units = unitstxt
#         sesamefile = open(fname,"r")  
#         sesamedata=sesamefile.readlines()
#         sesamefile.close()
#         nskip = 6 # skip standard header to get to the content of the 301 table
#         # num.density, num. temps
#         tmp = sesamedata[nskip][0:16]
#         dlen = float(tmp)
#         tmp = sesamedata[nskip][16:32]
#         tlen = float(tmp)
#         if (npy.mod((dlen*tlen*3.0+dlen+tlen+2.0),5.0) == 0):
#             neos = int((dlen*tlen*3.0+dlen+tlen+2.0)/5.0) 
#         else:
#             neos = int((dlen*tlen*3.0+dlen+tlen+2.0)/5.0) +1
#         #print(dlen,tlen,neos,len(sesamedata))
#         data = npy.zeros((neos,5),dtype=float)
#         for j in range(nskip,neos+nskip):
#             tmp3 = sesamedata[j]
#             tmp4 = list(tmp3.split())
#             if len(tmp4) < 5:
#                 lentmp4 = len(tmp4)
#                 data[j-nskip,0:lentmp4] = npy.asarray(tmp4[0:lentmp4])
#             else:
#                 data[j-nskip,:] = npy.asarray(tmp4)
#             #print(j,eosarr[j,:])
#         data=npy.resize(data,(neos*5))
#         self.ND  = data[0].astype(int)  # now fill the extEOStable class
#         self.NT  = data[1].astype(int)
#         self.rho = data[2:2+self.ND]
#         self.T   = data[2+self.ND : 2+self.ND+self.NT]
#         self.P   = data[2+self.ND+self.NT : 2+self.ND+self.NT+self.ND*self.NT
#                             ].reshape(self.NT,self.ND)
#         self.U   = data[2+self.ND+self.NT+self.ND*self.NT
#                             : 2+self.ND+self.NT+2*self.ND*self.NT
#                             ].reshape(self.NT,self.ND)
#         self.A   = data[2+self.ND+self.NT+2*self.ND*self.NT
#                             : 2+self.ND+self.NT+3*self.ND*self.NT
#                             ].reshape(self.NT,self.ND)
# 
# 
#     def loadextsesame(self, fname, unitstxt=None):
#         """Function for loading EXTENDED SESAME-STYLE EOS table output from ANEOS"""
#         data = ([])
#         if unitstxt is None:
#             self.units = 'Units: rho g/cm3, T K, P GPa, U MJ/kg, A MJ/kg, S MJ/K/kg, cs cm/s, cv MJ/K/kg, KPA flag. 2D arrays are (NT,ND).'
#         else:
#             self.units = unitstxt
#         sesamefile = open(fname,"r")  
#         sesamedata=sesamefile.readlines()
#         sesamefile.close()
#         nskip = 6 # skip standard header to get to the content of the 301 table
#         # num.density, num. temps
#         tmp = sesamedata[nskip][0:16]
#         dlen = float(tmp)
#         tmp = sesamedata[nskip][16:32]
#         tlen = float(tmp)
#         if (npy.mod((dlen*tlen*4.0+dlen+tlen+2.0),5.0) == 0):
#             neos = int((dlen*tlen*4.0+dlen+tlen+2.0)/5.0)
#         else:
#             neos = int((dlen*tlen*4.0+dlen+tlen+2.0)/5.0) +1
#         #print(dlen,tlen,neos,len(sesamedata))
#         data = npy.zeros((neos,5),dtype=float)
#         for j in range(nskip,neos+nskip):
#             tmp3 = sesamedata[j]
#             tmp4 = list(tmp3.split())
#             if len(tmp4) < 5:
#                 lentmp4 = len(tmp4)
#                 data[j-nskip,0:lentmp4] = npy.asarray(tmp4[0:lentmp4])
#             else:
#                 data[j-nskip,:] = npy.asarray(tmp4)        
#             #print(j,eosarr[j,:])
#         #print(data.shape)
#         data=npy.resize(data,(neos*5))
#         #print(data.shape)
#         self.ND  = data[0].astype(int)  # now fill the extEOStable class
#         self.NT  = data[1].astype(int)
#         self.rho = data[2:2+self.ND]
#         self.T   = data[2+self.ND : 2+self.ND+self.NT]
#         self.S   = data[2+self.ND+self.NT+0*self.ND*self.NT
#                             : 2+self.ND+self.NT+1*self.ND*self.NT
#                             ].reshape(self.NT,self.ND)
#         self.cs  = data[2+self.ND+self.NT+1*self.ND*self.NT
#                             : 2+self.ND+self.NT+2*self.ND*self.NT
#                             ].reshape(self.NT,self.ND)
#         self.cv  = data[2+self.ND+self.NT+2*self.ND*self.NT
#                             : 2+self.ND+self.NT+3*self.ND*self.NT
#                             ].reshape(self.NT,self.ND)
#         self.KPA = data[2+self.ND+self.NT+3*self.ND*self.NT
#                             : 2+self.ND+self.NT+4*self.ND*self.NT
#                             ].reshape(self.NT,self.ND)
# 
# 
#     def view(self, q='P', Tlow=None, Thigh=None, rholow=None, rhohigh=None):
#         """Function for printing values from EXTENDED SESAME-STYLE EOS table."""
#         if Tlow is None:
#             Tlow = self.T.min()
#         if Thigh is None:
#             Thigh = self.T.max()
#         if rholow is None:
#             rholow = self.rho.min()
#         if rhohigh is None:
#             rhohigh = self.rho.max()
#         print(self.units)
#         if q == 'P':
#             print('T:', self.T[npy.logical_and(self.T >= Tlow,self.T <= Thigh)])
#             print('rho:', self.rho[npy.logical_and(self.rho >= rholow,
#                                                    self.rho <= rhohigh)
#                                   ])
#             print('P:', (self.P[npy.logical_and(self.T >= Tlow,
#                                                 self.T<=Thigh)
#                                ])[:, npy.logical_and(self.rho >= rholow,
#                                                      self.rho <= rhohigh)
#                                  ])
#         if q == 'U':
#             print('T:', self.T[npy.logical_and(self.T >= Tlow,self.T <= Thigh)])
#             print('rho:', self.rho[npy.logical_and(self.rho >= rholow,
#                                                    self.rho <= rhohigh)
#                                   ])
#             print('U:', (self.U[npy.logical_and(self.T >= Tlow,
#                                                 self.T <= Thigh)
#                                ])[:, npy.logical_and(self.rho >= rholow,
#                                                      self.rho <= rhohigh)
#                                  ])
#         if q == 'A':
#             print('T:', self.T[npy.logical_and(self.T >= Tlow,self.T <= Thigh)])
#             print('rho:', self.rho[npy.logical_and(self.rho >= rholow,
#                                                    self.rho <= rhohigh)
#                                   ])
#             print('A:', (self.A[npy.logical_and(self.T >= Tlow,
#                                                 self.T <= Thigh)
#                                ])[:, npy.logical_and(self.rho >= rholow,
#                                                      self.rho <= rhohigh)
#                                  ])
#         if q == 'S':
#             print('T:', self.T[npy.logical_and(self.T >= Tlow,self.T <= Thigh)])
#             print('rho:', self.rho[npy.logical_and(self.rho
#                                    >= rholow,self.rho<=rhohigh)
#                                   ])
#             print('S:', (self.S[npy.logical_and(self.T >= Tlow,self.T <= Thigh)
#                                ])[:, npy.logical_and(self.rho >= rholow,
#                                                      self.rho <= rhohigh)
#                                  ])      
#         if q == 'cs':
#             print('T:', self.T[npy.logical_and(self.T >= Tlow,self.T <= Thigh)])
#             print('rho:', self.rho[npy.logical_and(self.rho >= rholow,
#                                                    self.rho<=rhohigh)])
#             print('cs:', (self.cs[npy.logical_and(self.T >= Tlow,
#                                                   self.T <= Thigh)
#                                  ])[:, npy.logical_and(self.rho >= rholow,
#                                                        self.rho <= rhohigh)
#                                    ])
#         if q == 'cv':
#             print('T:', self.T[npy.logical_and(self.T >= Tlow,self.T <= Thigh)])
#             print('rho:', self.rho[npy.logical_and(self.rho >= rholow,
#                                                    self.rho<=rhohigh)])
#             print('cv:', (self.cv[npy.logical_and(self.T >= Tlow,
#                                                   self.T <= Thigh)
#                                  ])[:, npy.logical_and(self.rho >= rholow,
#                                                        self.rho <= rhohigh)
#                                    ])
#         if q == 'KPA':
#             print('T:', self.T[npy.logical_and(self.T >= Tlow,self.T <= Thigh)])
#             print('rho:', self.rho[npy.logical_and(self.rho >= rholow,
#                                                    self.rho<=rhohigh)])
#             print('KPA:', (self.KPA[npy.logical_and(self.T >= Tlow,
#                                                   self.T <= Thigh)
#                                  ])[:, npy.logical_and(self.rho >= rholow,
#                                                        self.rho <= rhohigh)
#                                    ])
# 
#     def calchugoniot(self, r0=None, t0=None, pmax=None, writefilename=None, silent=False):
#         """Function for calculating a Hugoniot from EXTENDED SESAME-STYLE EOS table."""
#         if r0 is None:
#             return 'Must provide r0 and t0.'
#         if t0 is None:
#             return 'Must provide r0 and t0.'
#         if pmax is None:
#             pmax=1.E4 # GPa
#         self.hug.rho = []
#         self.hug.P = []
#         self.hug.T = []
#         self.hug.U = []
#         self.hug.S = []
#         self.hug.up = []
#         self.hug.us = []
#         self.hug.cs = []
#   
#         it0 = int(npy.round(npy.interp(t0,self.T,npy.arange(self.NT)))) # uses nearest value if t0 not in array
#         ir0 = int(npy.round(npy.interp(r0,self.rho,npy.arange(self.ND)))) # uses nearest value if r0 not in the array
#         p0  = self.P[it0,ir0] # GPa
#         #print(self.P[it0,ir0])
#         e0  = self.U[it0,ir0]#npy.interp(p0,self.P[it0,:],self.U[it0,:])
#         s0  = self.S[it0,ir0]#npy.interp(p0,self.P[it0,:],self.S[it0,:])
#         up0 = 0. # no initial particle velocity
#         us0 = self.cs[it0,ir0]/1.e5 # cm/s->km/s use sound velocity for initial
#         cs0 = self.cs[it0,ir0]/1.e5 # cm/s->km/s use sound velocity for initial
#         #print(ir0,it0,r0,t0,p0,e0,up0,us0)
#         self.hug.rho = npy.append(self.hug.rho, self.rho[ir0])
#         self.hug.P = npy.append(self.hug.P, p0)
#         self.hug.T = npy.append(self.hug.T, self.T[it0])
#         self.hug.U = npy.append(self.hug.U, e0)
#         self.hug.S = npy.append(self.hug.S, s0)
#         self.hug.up = npy.append(self.hug.up, up0)
#         self.hug.us = npy.append(self.hug.us, us0)
#         self.hug.cs = npy.append(self.hug.cs, cs0)
#         
#         #for iir in range(ir0+1,self.ND):
#         iir=ir0+1
#         pnew=p0
#         while pnew<pmax:
#             ediff =0.5*(self.P[it0::,iir]+p0)*(1./r0-1./self.rho[iir])+e0 -(self.U[it0::,iir])  # MJ/kg
#             #if not(pd.Series(ediff).is_monotonic_decreasing and pd.Series(ediff).is_unique):
#             if not ( npy.all(ediff[1:] < ediff[:-1])):
#                 if silent == False:
#                     print('bad ediff:', self.T[it0], self.rho[iir], pnew, tnew)
#             # npy.interp wants x values increasing
#             pnew = npy.interp(0.,npy.flip(ediff),npy.flip(self.P[it0::,iir]))
#             tnew = npy.interp(0.,npy.flip(ediff),npy.flip(self.T[it0::]))
#             enew = npy.interp(0.,npy.flip(ediff),npy.flip(self.U[it0::,iir]))
#             snew = npy.interp(0.,npy.flip(ediff),npy.flip(self.S[it0::,iir]))
#             upnew = npy.sqrt((pnew-p0)*(1./r0-1./self.rho[iir]))
#             usnew = (1./r0)*npy.sqrt((pnew-p0)/(1./r0-1./self.rho[iir]))
#             csnew = npy.interp(0.,npy.flip(ediff),npy.flip(self.cs[it0::,iir]))/1.E5 # km/s
#             #print(self.rho[iir],tnew,pnew,enew,upnew,usnew)
#             self.hug.rho = npy.append(self.hug.rho, self.rho[iir])
#             self.hug.P = npy.append(self.hug.P, pnew)
#             self.hug.T = npy.append(self.hug.T, tnew)
#             self.hug.U = npy.append(self.hug.U, enew)
#             self.hug.S = npy.append(self.hug.S, snew)
#             self.hug.up = npy.append(self.hug.up, upnew)
#             self.hug.us = npy.append(self.hug.us, usnew)
#             self.hug.cs = npy.append(self.hug.cs, csnew) # km/s
#             iir += 1
#         self.hug.NH=len(self.hug.P)
#         self.hug.units='units: T K, rho g/cm3, P GPa, U MJ/kg, S MJ/K/kg, Up km/s, Us km/s, cs km/s'
# 
#     def calcporoushugoniot(self, r0=None, t0=None, pmax=None, writefilename=None, r00=None):
#         """Function for calculating a Hugoniot from EXTENDED SESAME-STYLE EOS table."""
#         if r0 is None:
#             return 'Must provide r0 and t0.'
#         if t0 is None:
#             return 'Must provide r0 and t0.'
#         if pmax is None:
#             pmax=1.E4 # GPa
#         self.hug.rho = []
#         self.hug.P = []
#         self.hug.T = []
#         self.hug.U = []
#         self.hug.S = []
#         self.hug.up = []
#         self.hug.us = []
#         self.hug.cs = []
#   
#         it0 = int(npy.round(npy.interp(t0,self.T,npy.arange(self.NT)))) # uses nearest value if t0 not in array
#         ir0 = int(npy.round(npy.interp(r0,self.rho,npy.arange(self.ND)))) # uses nearest value if r0 not in the array
#         ir00 = int(npy.round(npy.interp(r00,self.rho,npy.arange(self.ND)))) # uses nearest value if r0 not in the array
#         p0  = self.P[it0,ir0] # GPa
#         #print(self.P[it0,ir0])
#         e0  = self.U[it0,ir0]#npy.interp(p0,self.P[it0,:],self.U[it0,:])
#         s0  = self.S[it0,ir0]#npy.interp(p0,self.P[it0,:],self.S[it0,:])
#         up0 = 0. # no initial particle velocity
#         us0 = self.cs[it0,ir0]/1.e5 # cm/s->km/s use sound velocity for initial
#         cs0 = self.cs[it0,ir0]/1.e5 # cm/s->km/s use sound velocity for initial
#         #print(ir0,it0,r0,t0,p0,e0,up0,us0)
#         self.hug.rho = npy.append(self.hug.rho, self.rho[ir00])
#         self.hug.P = npy.append(self.hug.P, p0)
#         self.hug.T = npy.append(self.hug.T, self.T[it0])
#         self.hug.U = npy.append(self.hug.U, e0)
#         self.hug.S = npy.append(self.hug.S, s0)
#         self.hug.up = npy.append(self.hug.up, up0)
#         self.hug.us = npy.append(self.hug.us, us0)
#         self.hug.cs = npy.append(self.hug.cs, cs0)
#         
#         #for iir in range(ir0+1,self.ND):
#         iir=ir00+1
#         pnew=p0
#         while pnew<pmax:
# #            ediff =0.5*(self.P[it0::,iir]+p0)*(1./r0-1./self.rho[iir])+e0 -(self.U[it0::,iir])  # MJ/kg
#             ediff =0.5*(self.P[it0::,iir]+p0)*(1./r00-1./self.rho[iir])+e0 -(self.U[it0::,iir])  # MJ/kg
#             # npy.interp wants x values increasing
#             pnew = npy.interp(0.,npy.flip(ediff),npy.flip(self.P[it0::,iir]))
#             tnew = npy.interp(0.,npy.flip(ediff),npy.flip(self.T[it0::]))
#             enew = npy.interp(0.,npy.flip(ediff),npy.flip(self.U[it0::,iir]))
#             snew = npy.interp(0.,npy.flip(ediff),npy.flip(self.S[it0::,iir]))
#             upnew = npy.sqrt((pnew-p0)*(1./r00-1./self.rho[iir]))
#             usnew = (1./r00)*npy.sqrt((pnew-p0)/(1./r00-1./self.rho[iir]))
#             csnew = npy.interp(0.,npy.flip(ediff),npy.flip(self.cs[it0::,iir]))/1.E5 # km/s
#             #print(self.rho[iir],tnew,pnew,enew,upnew,usnew)
#             self.hug.rho = npy.append(self.hug.rho, self.rho[iir])
#             self.hug.P = npy.append(self.hug.P, pnew)
#             self.hug.T = npy.append(self.hug.T, tnew)
#             self.hug.U = npy.append(self.hug.U, enew)
#             self.hug.S = npy.append(self.hug.S, snew)
#             self.hug.up = npy.append(self.hug.up, upnew)
#             self.hug.us = npy.append(self.hug.us, usnew)
#             self.hug.cs = npy.append(self.hug.cs, csnew) # km/s
#             iir += 1
#         self.hug.NH=len(self.hug.P)
#         self.hug.units='units: T K, rho g/cm3, P GPa, U MJ/kg, S MJ/K/kg, Up km/s, Us km/s, cs km/s'
# 
#         if writefilename:
#             print('Writing Hugoniot to file: ',writefilename)
#             hugoniotfile = open(writefilename,"w")  
#             hugoniotfile.writelines('  Hugoniot \n') 
#             hugoniotfile.writelines('  Temperature,    Density,        Pressure,       Int. Energy,    Sp. Entropy,    Part. Vel.,     Shock Vel. \n') 
#             hugoniotfile.writelines('  K,              g/cm3,          GPa,            MJ/kg,          MJ/K/kg,        km/s,           km/s\n') 
#             for iih in range(0,self.hug.NH):
#                 hugoniotfile.write("%14.6e, %14.6e, %14.6e, %14.6e, %14.6e, %14.6e, %14.6e\n" % (
#                     self.hug.T[iih],self.hug.rho[iih],self.hug.P[iih],self.hug.U[iih],self.hug.S[iih],self.hug.up[iih],self.hug.us[iih]))
#             hugoniotfile.close() 
# 
#     def writestdsesame(self, writestdsesfname=None):
#         """Write standard Header-201-301 SESAME EOS TABULAR EOS FILE"""
#         # write a standard SESAME ascii file
#         #     WRITE STANDARD Header-201-301 SESAME FILE
#         #     WRITE SESAME 301 TABLE CONTAINS P, E, HFE
#         #sesfile = open("NEW-SESAME-STD-NOTENSION.EOSTXT","w")  
#         if writestdsesfname is None:
#             print('Please provide a file name.')
#             exit(0)
#         sesfile = open(writestdsesfname,"w")  
#         #     WRITE SESAME HEADER INFORMATION: EOS matid number, number of words in section
#         #     could input matid, date, version with the grid
#         # these parameters are set in the cell above that sets up the grid for ANEOS
#         # THEY SHOULD MATCH.......
#         # These variables are needed for the standard table output
#         NWDS=9
#         SESNTABLES=2.0
#         TABLE1 = 201.0
#         TABLE2 = 301.0
#         #     5 entries in 201 table
#         SESNWDS1=5.0
#         #     Number of entries in STANDARD 301 table: 3 variables at each T,rho point
#         SESNWDS2=2.+self.ND+self.NT+self.ND*self.NT*3.
#         #     HEADER SECTION
#         #sesfile.write("%14.6e, %14.6e, %14.6e, %14.6e, %14.6e, %14.6e, %14.6e\n" % (antarr[iit],rnew,pnew,enew,snew,upnew,usnew))
#         sesfile.write(" INDEX      MATID ={:7d}    NWDS = {:8d}\n".format(int(self.MATID), int(NWDS)))
#         sesfile.write("{:16.8e}{:16.8e}{:16.8e}{:16.8e}{:16.8e}\n".format(self.MATID, self.DATE, self.DATE, self.VERSION, SESNTABLES))
#         sesfile.write("{:16.8e}{:16.8e}{:16.8e}{:16.8e}\n".format(TABLE1, TABLE2, SESNWDS1, SESNWDS2))
#         # 201 SECTION
#         sesfile.write(" RECORD     TYPE ={:5d}     NWDS = {:8d}\n".format(int(TABLE1),int(SESNWDS1)))
#         sesfile.write("{:16.8e}{:16.8e}{:16.8e}{:16.8e}{:16.8e}\n".format(self.FMN, self.FMW, self.R0REF, self.K0REF, self.T0REF))
#         sesfile.write(" RECORD     TYPE ={:5d}     NWDS = {:8d}\n".format(int(TABLE2),int(SESNWDS2)))
#         sesfile.write("{:16.8e}{:16.8e}".format(self.ND, self.NT))
#         STYLE=2
#         #     density array g/cm3
#         for k in range(0, int(self.ND)):
#             sesfile.write("{:16.8e}".format(self.rho[k]))
#             STYLE=STYLE+1
#             if (npy.mod(STYLE,5) == 0):
#                 sesfile.write("\n")
#         #     temperature array K
#         for j in range(0, int(self.NT)):
#             sesfile.write("{:16.8e}".format(self.T[j]))
#             STYLE=STYLE+1
#             if (npy.mod(STYLE,5) == 0):
#                 sesfile.write("\n")
#         #  pressure array GPa P[tempindex,dindex]
#         for j in range(0,int(self.NT)):
#             for k in range(0,int(self.ND)):
#                 sesfile.write("{:16.8e}".format(self.P[j,k]))
#                 STYLE=STYLE+1
#                 if (npy.mod(STYLE,5) == 0):
#                     sesfile.write("\n")
#         #  specific internal energy array MJ/kg U[tempindex,dindex]
#         for j in range(0,int(self.NT)):
#             for k in range(0,int(self.ND)):
#                 sesfile.write("{:16.8e}".format(self.U[j,k]))
#                 STYLE=STYLE+1
#                 if (npy.mod(STYLE,5) == 0):
#                     sesfile.write("\n")
#         # Helmholtz free energy array in MJ/kg A[tempindex,dindex]
#         for j in range(0,int(self.NT)):
#             for k in range(0,int(self.ND)):
#                 sesfile.write("{:16.8e}".format(self.A[j,k]))
#                 STYLE=STYLE+1
#                 if (npy.mod(STYLE,5) == 0):
#                     sesfile.write("\n")
#         # close the SESAME TABLE FILE
#         sesfile.close() 
#         print('Done writing the STD SESAME 301 notension table to local directory: ',writestdsesfname)
# 
#     def writemdqsesame(self, writemdqsesfname=None):
#         """Function to write a sesame 301-style ascii file with the MDQ variable"""
#         if writemdqsesfname is None:
#             print('Please provide a file name.')
#             exit(0)
#         sesfile = open(writemdqsesfname,"w")  
#         sesfile.write("{:16.8e}{:16.8e}".format(self.ND, self.NT))
#         STYLE=2
#         #     density array g/cm3
#         for k in range(0, int(self.ND)):
#             sesfile.write("{:16.8e}".format(self.rho[k]))
#             STYLE=STYLE+1
#             if (npy.mod(STYLE,5) == 0):
#                 sesfile.write("\n")
#         #     temperature array K
#         for j in range(0, int(self.NT)):
#             sesfile.write("{:16.8e}".format(self.T[j]))
#             STYLE=STYLE+1
#             if (npy.mod(STYLE,5) == 0):
#                 sesfile.write("\n")
#         #  MDQ Flag[tempindex,dindex]
#         for j in range(0,int(self.NT)):
#             for k in range(0,int(self.ND)):
#                 sesfile.write("{:16.8e}".format(self.MDQ[j,k]))
#                 STYLE=STYLE+1
#                 if (npy.mod(STYLE,5) == 0):
#                     sesfile.write("\n")
#         # close the SESAME TABLE FILE
#         sesfile.close() 
#         print('Done writing the MDQ Flag as a 301-style table to local directory: ',writemdqsesfname)
# 
#     def loadaneos(self, aneosinfname=None, aneosoutfname=None, silent=False):
#         """Function for reading in ANEOS INPUT and OUTPUT FILE DATA into EOS structure."""
#         if aneosinfname is None:
#             return 'Must provide input file name.'
#         if aneosoutfname is None:
#             return 'Must provide output file name.'
#         # function to gather data from ANEOS input and output files
#         # SESAME FILE HEADER INFORMATION MUST BE LOADED INTO THE EOS STRUCTURE BEFORE CALLING THIS FUNCTION
#         #
#         # READ IN ANEOS INPUT FILE
#         aneosinputfile = open(aneosinfname,"r")  
#         testin=aneosinputfile.readlines()   # read in the whole ascii file at once because this is fatser
#         aneosinputfile.close()
#         # gather EOS information from the ANEOS.OUTPUT file
#         aneosoutputfile = open(aneosoutfname,"r")  
#         testout=aneosoutputfile.readlines() # read everything in at once because this is faster
#         aneosoutputfile.close()
#         if silent == False:
#             print('Done loading ANEOS files.')
# 
#         # THIS CODE PARSES THE ANEOS.OUTPUT FILE INTO ARRAYS FOR USE IN PLOTTING/USING THE EOS
#         if silent == False:
#             print('ANEOS WAS CALLED WITH THE FOLLOWING INPUT, LOADED FROM FILE ',aneosinfname)
#         # Gather parameters for the gamma function while printing the ANEOS INPUT FILE
#         aneoscount=1
#         C60=0.
#         C61=0.
#         betagamma=0.
#         for i in npy.arange(len(testin)):
#             if testin[i].find('ANEOS') == 0:
#                 if aneoscount<9:
#                     if silent == False:
#                         print(' '+testin[i-3],testin[i-2],testin[i-1],testin[i])
#                     aneoscount=aneoscount+1
#                 else:
#                     if silent == False:
#                         print(' '+testin[i])
#                 if testin[i].find('ANEOS2') == 0:
#                     tmp=testin[i]
#                     nelem=int(tmp[10:20])
#                     #print('nelem=',nelem)
#                     rho0=float(tmp[30:40])
#                     #print('rho0=',rho0)
#                     gamma0=float(tmp[70:80])
#                     #print('gamma0=',gamma0)
#                     theta0=float(tmp[80:90])
#                 if testin[i].find('ANEOS3') == 0:
#                     tmp=testin[i]
#                     C24=float(tmp[20:30])/3.
#                     #print('C24=',C24)
#                 if testin[i].find('ANEOS5') == 0:
#                     tmp=testin[i]
#                     C60=float(tmp[60:70])
#                     C61=float(tmp[70:80])
#                     #print('C60=',C60)
#                 if testin[i].find('ANEOS7') == 0:
#                     tmp=testin[i]
#                     betagamma=float(tmp[70:80])
# 
#         # some checks
#         if rho0 != self.R0REF:
#             print('WARNING: rho0 does not match. STOPPING THIS NOTEBOOK.')
#             assert(False) # just a way to stop the notebook
# 
#         # GUESS A BIG ARRAY SIZE FOR THE PHASE BOUNDARIES AND HUGONIOT IN ANEOS.OUTPUT
#         # the melt curve, vapor curve and Hugoniot curves are not fixed length outputs
#         nleninit=300
#         meltcurve = 0
# 
#         if silent == False:
#             print('READING DATA FROM ANEOS OUTPUT FILE ',aneosoutfname)
# 
#         # Read in data from the ANEOS.OUTPUT FILE
#         imc = -1 # flag for no melt curve in the model
#         for i in npy.arange(len(testout)):
#             if testout[i].find('  Data for ANEOS number') == 0:
#                 tmp = testout[i+2][0:50]
#                 eosname = tmp.strip()
#             if testout[i] == '  TWO-PHASE BOUNDARIES\n':
#                 nvc = nleninit
#                 ivc = i
#                 vcarrtmp = npy.zeros((nvc,12),dtype=float)
#                 flag=0
#                 j=0
#                 while flag == 0:
#                     if testout[j+i+4].find(' anphas') == 0:
#                         print(testout[j+i+4])
#                         vcarrtmp[j,:]=vcarrtmp[j-1,:]
#                         j=j+1
#                     else:
#                         tmp=str.replace(testout[j+i+4],'D','E')
#                         tmp3 = tmp[0:157]
#                         tmp4 = list(tmp3.split())
#                         if (tmp4[3].find('E') == -1): # if the number is formatted badly, stop reading in the vapor curve, e.g. 1.00000-321
#                             print('Stopped reading in the vapor curve because of badly formatted numbers. Stopped at this line:')
#                             print(tmp3)
#                             flag=1
#                         else:
#                             if (len(tmp4) >0) and (float(tmp4[3]) > 0) and (float(tmp4[4]) > 0): # stop if the pressures become negative on the vapor curve
#                                 tmp5 = npy.asarray(tmp4)
#                                 vcarrtmp[j,:] = tmp5[:]
#                                 j=j+1
#                             else:
#                                 flag=1
#                 vcarr = npy.zeros((j,12),dtype=float)
#                 vcarr[:,:] = vcarrtmp[0:j,:]
#             if testout[i] == ' LIQUID/SOLID PHASE CURVE\n':
#                 nmc = nleninit
#                 imc = i
#                 meltcurve=1
#                 mcarrtmp = npy.zeros((nmc,11),dtype=float)
#                 flag=0
#                 j=0
#                 while flag == 0:
#                     tmp  = str.replace(testout[j+i+5],'D','E')
#                     tmp3 = tmp[0:132]
#                     tmp4 = list(tmp3.split())
#                     if len(tmp4) > 0:
#                         tmp5 = npy.asarray(tmp4)
#                         mcarrtmp[j,:] = tmp5[:]
#                         j=j+1
#                     else:
#                         flag=1
#                 mcarr = npy.zeros((j,11),dtype=float)
#                 mcarr[:,:] = mcarrtmp[0:j,:]
#             if testout[i] == '   HUGONIOT\n':
#                 nhc = nleninit
#                 ihc = i
#                 hcarrtmp = npy.zeros((nhc,9),dtype=float)
#                 flag=0
#                 j=0
#                 while flag == 0:
#                     tmp=str.replace(testout[j+i+5],'D','E')
#                     tmp3 = tmp[0:109]
#                     tmp4 = list(tmp3.split())
#                     if len(tmp4) > 0:
#                         tmp4[3]='0.0' # this column often gives problems with exponential notation so don't read it
#                         tmp5 = npy.asarray(tmp4)
#                         hcarrtmp[j,:] = tmp5[:]
#                         j=j+1
#                     else:
#                         flag=1
#                 hcarr = npy.zeros((j,9),dtype=float)
#                 hcarr[:,:] = hcarrtmp[0:j,:]
# 
#         # UPDATE THE MAIN EOS STRUCTURE WITH GATHERED INFORMATION
#         # Add variables needed to calculate the ANEOS gamma function
#         self.gamma0  = gamma0
#         self.theta0  = theta0
#         self.C24     = C24
#         self.C60     = C60
#         self.C61     = C61
#         self.beta    = betagamma
#         #
#         # ANEOS.OUTPUT UNITS ARE NOT THE SAME AS THE SESAME TABLE!
#         # add the vapor curve to this EOS object extracted from the ANEOS.OUTPUT
#         #  TWO-PHASE BOUNDARIES
#         #       T         RHOLIQ        RHOVAP        PLIQ         PVAP        ELIQ         EVAP         SLIQ         SVAP        GLIQ         GVAP         PSILIQ      PSIVAP         NTY
#         #       K         kg/m**3       kg/m**3       GPa          GPa         J/kg         J/kg        J/kg-K       J/kg-K       J/kg         J/kg
#         tmp = vcarr.shape
#         #put vapor curve information in nicely named structure
#         self.vc.NT  = tmp[0]
#         self.vc.T   = vcarr[:,0] # K
#         self.vc.rl  = vcarr[:,1]/1.E3 # g/cm3
#         self.vc.rv  = vcarr[:,2]/1.E3 # g/cm3
#         self.vc.Pl  = vcarr[:,3] # GPa
#         self.vc.Pv  = vcarr[:,4] # GPa
#         self.vc.Ul  = vcarr[:,5]/1.E6 # MJ/kg
#         self.vc.Uv  = vcarr[:,6]/1.E6 # MJ/kg
#         self.vc.Sl  = vcarr[:,7]/1.E6 # MJ/K/kg
#         self.vc.Sv  = vcarr[:,8]/1.E6 # MJ/K/kg
#         self.vc.Gl  = vcarr[:,9]/1.E6 # MJ/kg
#         self.vc.Gv  = vcarr[:,10]/1.E6 # MJ/kg
#         self.vc.units = 'T K, rho g/cm3, P GPa, U MJ/kg, S MJ/K/kg, G MJ/kg'
#         # npy.interp wants increasing x values
#         self.onebar.Tvap = npy.interp(1.E-4,npy.flipud(self.vc.Pl),npy.flipud(self.vc.T)) # extract boiling point temperature at 1 bar, K
#         self.onebar.Siv  = npy.interp(1.E-4,npy.flipud(self.vc.Pl),npy.flipud(self.vc.Sl)) # extract liquid sp. entropy at 1 bar, MJ/K/kg
#         self.onebar.Scv  = npy.interp(1.E-4,npy.flipud(self.vc.Pl),npy.flipud(self.vc.Sv)) # extract vapor sp. entropy at 1 bar, MJ/K/kg
#         self.onebar.rhoiv  = npy.interp(1.E-4,npy.flipud(self.vc.Pl),npy.flipud(self.vc.rl)) # extract liquid density at 1 bar, g/cm3
#         self.onebar.rhocv  = npy.interp(1.E-4,npy.flipud(self.vc.Pl),npy.flipud(self.vc.rv)) # extract vapor density at 1 bar, g/cm3
#         #
#         # add the ANEOS Hugoniot to this EOS object extracted from the ANEOS.OUTPUT
#         #      RHO          T           P          PC           E           S           V           U       RHO/RHOO  #IT  STATE
#         #    kg/m**3        K          GPa        GPa          J/kg      J/kg-K       km/sec      km/sec
#         #self.anhug.all = hcarr  # 2D array of Hugoniot variables
#         tmp = hcarr.shape
#         self.anhug.ND = tmp[0] # number of density points on the Hugoniot
#         self.anhug.rho = hcarr[:,0]/1.E3 # g/cm3
#         self.anhug.T   = hcarr[:,1] # K
#         self.anhug.P   = hcarr[:,2] # GPa
#         self.anhug.U   = hcarr[:,4]/1.E6 # MJ/kg
#         self.anhug.S   = hcarr[:,5]/1.E6 # MJ/K/kg
#         self.anhug.us  = hcarr[:,6] # km/s
#         self.anhug.up  = hcarr[:,7] # km/s
#         self.anhug.units = 'vars: rho g/cm3, T K, P GPa, U MJ/kg, S MJ/K/kg, Us km/s, Up km/s'
#         #
#         # Add melt curve to EOS objects if available
#         # LIQUID/SOLID PHASE CURVE
#         #       T         RLIQ       RSOLID      PLIQ       PSOLID      ELIQ        ESOLID       SLIQ       SOLID        GLIQ       GSOLID        #ITER
#         #       K        kg/m**3     kg/m**3      GPa         GPa       J/kg         J/kg        J/kg-K     J/kg-K       J/kg        J/kg
#         if meltcurve == 1:
#             # put the melt curve information in nicely named structure
#             tmp=mcarr.shape
#             self.mc.NT  = tmp[0]
#             self.mc.T   = mcarr[:,0] # K
#             self.mc.rl  = mcarr[:,1]/1.E3 # g/cm3
#             self.mc.rs  = mcarr[:,2]/1.E3 # g/cm3
#             self.mc.Pl  = mcarr[:,3] # GPa
#             self.mc.Ps  = mcarr[:,4] # GPa
#             self.mc.Ul  = mcarr[:,5]/1.E6 # MJ/kg
#             self.mc.Us  = mcarr[:,6]/1.E6 # MJ/kg
#             self.mc.Sl  = mcarr[:,7]/1.E6 # MJ/K/kg
#             self.mc.Ss  = mcarr[:,8]/1.E6 # MJ/K/kg
#             self.mc.units = 'T K, rho g/cm3, P GPa, U MJ/kg, S MJ/K/kg'
#             # NOTE THAT TRIPLE POINT AND VAPOR CURVE SOLUTIONS DO NOT ALWAYS MATCH PERFECTLY AT THE TRIPLE POINT
#             tmp = npy.where(mcarr[:,3] > 0.)[0] # find the triple point first entry with positive pressure
#             self.tp.T = mcarr[tmp[0],0] # K
#             self.tp.P = npy.interp(self.tp.T,npy.flipud(self.vc.T),npy.flipud(self.vc.Pv))   # this has trouble for forsterite; use the vapor size of the VC mcarr[:,3] # GPa
#             self.tp.Sim  = mcarr[tmp[0],8]/1.E6 # extract solid sp. entropy at tp, MJ/K/kg
#             self.tp.Scm  = mcarr[tmp[0],7]/1.E6 # extract liquid sp. entropy at tp, MJ/K/kg
#             # npy.interp wants x values increasing
#             self.tp.Siv  = npy.interp(self.tp.T,npy.flipud(self.vc.T),npy.flipud(self.vc.Sl)) # extract liquid sp. entropy at tp, MJ/K/kg
#             self.tp.Scv  = npy.interp(self.tp.T,npy.flipud(self.vc.T),npy.flipud(self.vc.Sv)) # extract vapor sp. entropy at tp, MJ/K/kg
#             self.tp.rhol = mcarr[tmp[0],1]/1.E3 # extract liquid density at tp, g/cm3
#             self.tp.rhos = mcarr[tmp[0],2]/1.E3 # extract solid density at tp, g/cm3
#             self.tp.rhov = npy.interp(self.tp.T,npy.flipud(self.vc.T),npy.flipud(self.vc.rv)) # extract vapor density at tp, g/cm3
#             self.tp.units = 'T K, P GPa, S MJ/K/kg, rho g/cm3'
#             # Extract melting point
#             self.onebar.Tmelt = npy.interp(1.E-4,self.mc.Pl,self.mc.T) # extract melting point temperature at 1 bar, K
#             self.onebar.Sim   = npy.interp(1.E-4,self.mc.Pl,self.mc.Ss) # extract liquid sp. entropy at 1 bar BP, MJ/K/kg
#             self.onebar.Scm   = npy.interp(1.E-4,self.mc.Pl,self.mc.Sl) # extract vapor sp. entropy at 1 bar BP, MJ/K/kg
#             self.onebar.rhoim   = npy.interp(1.E-4,self.mc.Pl[3::],self.mc.rs[3::]) # extract solid density at 1 bar MP, MJ/K/kg
#             self.onebar.rhocm   = npy.interp(1.E-4,self.mc.Pl[3::],self.mc.rl[3::]) # extract liquid density at 1 bar MP, MJ/K/kg
#         # put the data for the critical point in the EOS structure for easy access
#         self.cp.T   = vcarr[0,0] # K
#         self.cp.rho = vcarr[0,1]/1.E3 # g/cm3
#         self.cp.P   = vcarr[0,3] # GPa
#         self.cp.U   = vcarr[0,5]/1.E6 # MJ/kg 
#         self.cp.S   = vcarr[0,7]/1.E6 # MJ/K/kg
#         self.cp.units = 'T K, rho g/cm3, P GPa, U MJ/kg, S MJ/K/kg'
#         #------------------------------------------------------------------




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


