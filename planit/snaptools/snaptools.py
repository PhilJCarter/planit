"""
   planit snapshot and header classes
"""

from ..globaldefs import *
from .. import utils
from .. import eos
from . import io

import os
import numpy as npy
import scipy
import h5py
#import struct


class SnapHeader:
    """
    Class for Gadget/Swift snapshot header
    """ 
    def __init__(self, t=0.0, nfiles=1, ent=1):
        """
        Creates SnapHeader object
        """
        self.npart = npy.zeros(6).astype(int)
        self.mass = npy.zeros(6)
        self.time = t
        self.redshift = 0.0
        self.flag_sfr = 0
        self.flag_feedbacktp = 0
        self.npartTotal = npy.zeros(6).astype(int)
        self.flag_cooling = 0
        self.num_files = nfiles
        self.BoxSize = 0.0 #npy.zeros(3)
        self.Omega0 = 0.0
        self.OmegaLambda = 0.0
        self.HubbleParam = 1.0
        self.flag_stellarage = 0
        self.flag_metals = 0
        self.nallhw = npy.zeros(6).astype(int)
        self.flag_entr_ics = ent


class Snapshot:
    """
    Gadget/Swift snapshot class
    
    Includes header and gas particle data, with functions for
    reading and writing snapshots.
   
    load() -- load Gadget snapshot data
    remove() -- remove particle from snapshot
    write() -- save snapshot
    ic_from_seagen() -- create snapshot from seagen particle planet
    combine() -- create snapshot from two other snapshots
    G2_to_swift() -- output Gadget2 format snapshot in swift hdf5 format
    identify() -- determine material types
    summary() -- print basic info
    eq_test() -- test for equilibration (rms v below threshold multiple of escape v)
    bound_mass() -- calculate remnant mass(es)
    calc_phase() -- calculate material phases
    calc_vap_frac() -- calculate vapour fractions of particles
    """
    
    def __init__(self):
        """
        Creates Snapshot object
        """
        self.header = SnapHeader()
        self.N = 0              # number of SPH particles
        self.initarrays()
        # Extras
        self.file = None        # name of file read from
        self.inclthermo = False # read/calculate extra thermodynamic variables
        

    def initarrays(self):
        """
        Initialises particle attributes
        """
        self.pos = npy.empty(0)  # position vector
        self.x = npy.empty(0)
        self.y = npy.empty(0)
        self.z = npy.empty(0)
        self.vel = npy.empty(0)  # velocity vector
        self.vx = npy.empty(0)
        self.vy = npy.empty(0)
        self.vz = npy.empty(0)
        self.id = npy.empty(0)   # particle ID
        self.m = npy.empty(0)    # mass
        self.S = npy.empty(0)    # entropy
        self.rho = npy.empty(0)  # density
        self.hsml = npy.empty(0) # smoothing length
        self.pot = npy.empty(0)  # potential
        # Thermo extension
        self.P = npy.empty(0)    # pressure
        self.T = npy.empty(0)    # temperature
        self.U = npy.empty(0)    # internal energy
        self.cs = npy.empty(0)   # sound speed
        # extras
        self.materialIDs = None  # particle material ID
        self.rem = None          # ID of remnant particle belongs to
        self.vapfrac = None      # vapour fraction
        self.meltfrac = None     # particle melt fraction
        self.phase = None        # particle phase number
        # not used:
        #self.accel = npy.empty(0)  # acceleration
        #self.dt = npy.empty(0)  # time step
        

    def __getattribute__(self, attr):
        """
        Automatically attempts read from file if attribute is accessed but not already loaded
        """
        if attr in ['x','y','z','vx','vy','vz','pos','vel','id','m','S','rho','hsml','pot','P','T','U','cs'] and self.file:
            if len(super().__getattribute__(attr))==0 and self.file:
                self.load(self.file, headonly=False, compress=False, thermo=self.inclthermo, loadprops=[attr,])
        elif attr in ['rem', 'bnd']:
            if super().__getattribute__(attr) is None and self.file:
                if h5py.is_hdf5(self.file):
                    with h5py.File(self.file,'r') as f:
                        if 'RemnantIDs' in f['PartType0'].keys():   
                            self.load(self.file, headonly=False, compress=False, thermo=self.inclthermo, loadprops=[attr,])
                        #else:
                        #    #self.bound_mass()
                        #    print('run bound_mass()')
                else:
                    if os.path.exists(str(self.file)+'_rem.txt'):
                        self.load(self.file, headonly=False, compress=False, thermo=self.inclthermo, loadprops=[attr,])
                    #else:
                    #    #self.bound_mass()
                    #    print('run bound_mass()')
        elif attr in ['vapfrac','meltfrac','phase']:
            if super().__getattribute__(attr) is None and len(self.S) > 0:
                self.calc_phase()
        return super().__getattribute__(attr)

    
    def freedata(self):
        """
        Clear all loaded data
        """
        self.initarrays()


    def ensure_matIDs(self,mats):
        """
        If materialIDs missing, add correct material IDs
        """
        eosIDs = [eos.select(mat).womaID for mat in mats]
        if npy.ndim(self.materialIDs) < 1:
            self.materialIDs = npy.choose( (self.id/GADGET_EOS_OFFSET).astype(int), eosIDs )
                
        
    def load(self, fname, headonly = False, thermo=False, compress=False, mats=[402,400], loadprops=['all',]):
        """
        Loads snapshot data from file
        """
        io.load_snapshot(self, fname, headonly=headonly, thermo=thermo, compress=compress, mats=mats, loadprops=loadprops)
    

    def ic_from_seagen(self, partplanet, thermo=False, init_h=100e5):
        io.load_seagen(self, partplanet, thermo, init_h)


    def combine(self, body1, body2, bidoffset=PROJ_ID_OFFSET, thermo=False, box=0.0):
        """
        Assign snapshot particle data by combining two other Snapshots
        """
        #HEADER
        self.header.npart = body1.header.npart+body2.header.npart
        self.header.mass = npy.array([0., 0., 0., 0., 0., 0.])
        self.header.time = 0.
        self.header.redshift= 0.
        self.header.flag_sfr = self.header.flag_feedbacktp = self.header.flag_cooling = 0
        self.header.npartTotal = self.header.npart
        self.header.num_files = 1
        if box>0.0:
            self.header.BoxSize = box
        else:
            self.header.BoxSize = body1.header.BoxSize
        self.header.Omega0 = self.header.OmegaLambda = 0.0
        self.header.HubbleParam = 1.0
        self.header.flag_stellarage = self.header.flag_metals = 0
        self.header.nallhw = npy.array([0, 0, 0, 0, 0, 0])
        if body1.header.flag_entr_ics[0] != body2.header.flag_entr_ics[0]:
            raise ValueError("Entropy IC flags must match!", body1.header.flag_entr_ics[0], body2.header.flag_entr_ics[0])
            
        self.header.flag_entr_ics = body1.header.flag_entr_ics

        self.N = self.header.npart[0]
        
        if self.N/2.2 > bidoffset:
            print('WARNING: low body ID offset, N =', self.N,' body ID offset =', bidoffset)

        #PARTICLE DATA
        self.x = npy.append(body1.x,body2.x)
        self.y = npy.append(body1.y,body2.y)
        self.z = npy.append(body1.z,body2.z)
        self.pos = npy.array((self.x, self.y, self.z))
        self.pos = self.pos.T
        
        self.vx = npy.append(body1.vx,body2.vx)
        self.vy = npy.append(body1.vy,body2.vy)
        self.vz = npy.append(body1.vz,body2.vz)
        self.vel = npy.array((self.vx, self.vy, self.vz))
        self.vel = self.vel.T

        self.id = npy.append(body1.id,body2.id+bidoffset)
        self.m = npy.append(body1.m,body2.m)
        self.S = npy.append(body1.S,body2.S)
        self.rho = npy.append(body1.rho,body2.rho)
        self.pot = npy.append(body1.pot,body2.pot)
        self.hsml = npy.append(body1.hsml,body2.hsml)
        if npy.ndim(body1.materialIDs) > 0 and npy.ndim(body2.materialIDs) > 0:
            self.materialIDs = npy.append(body1.materialIDs,body2.materialIDs)
        if len(body1.U)>0 and len(body2.U)>0:
            self.U = npy.append(body1.U,body2.U)
        if len(body1.P)>0 and len(body2.P)>0:
            self.P = npy.append(body1.P,body2.P)
        

    def remove(self, pid):
        """
        Remove particle from Snapshot
        """
        if pid not in self.id:
            raise ValueError('Particle:', pid, 'not found in Snapshot')
        self.header.npart[0] = self.header.npart[0]-1
        self.N = self.header.npart[0]
        
        self.x = npy.delete(self.x, npy.where(self.id == pid))
        self.y = npy.delete(self.y, npy.where(self.id == pid))
        self.z = npy.delete(self.z, npy.where(self.id == pid))
        self.vx = npy.delete(self.vx, npy.where(self.id == pid))
        self.vy = npy.delete(self.vy, npy.where(self.id == pid))
        self.vz = npy.delete(self.vz, npy.where(self.id == pid))
        self.m = npy.delete(self.m, npy.where(self.id == pid))
        if self.materialIDs is not None:
            self.materialIDs = npy.delete(self.materialIDs, npy.where(self.id == pid))
        if len(self.S) > 0:
            self.S = npy.delete(self.S, npy.where(self.id == pid))
        self.rho = npy.delete(self.rho, npy.where(self.id == pid))
        self.hsml = npy.delete(self.hsml, npy.where(self.id == pid))
        self.pot = npy.delete(self.pot, npy.where(self.id == pid))
        if len(self.P) > 0:
            self.P = npy.delete(self.P, npy.where(self.id == pid))
        if len(self.T) > 0:
            self.T = npy.delete(self.T, npy.where(self.id == pid))
        if len(self.U) > 0:
            self.U = npy.delete(self.U, npy.where(self.id == pid))
        if len(self.cs) > 0:
            self.cs = npy.delete(self.cs, npy.where(self.id == pid))
#        if type(self.accel) != int:
#            self.ax = npy.delete(self.ax, npy.where(self.id == pid))
#            self.ay = npy.delete(self.ay, npy.where(self.id == pid))
#            self.az = npy.delete(self.az, npy.where(self.id == pid))
#            self.dt = npy.delete(self.dt, npy.where(self.id == pid))
        if self.rem is not None:
            self.rem = npy.delete(self.rem, npy.where(self.id == pid))
        if super().__getattribute__('vapfrac') is not None:
            # calc_phase would be triggered by a normal lookup and requires id to already be shortened, 
            # but id needs to be last attribute in order for particle ID matching. Also don't really want 
            # to calculate in this case anyway
            self.vapfrac = npy.delete(self.vapfrac, npy.where(self.id == pid))
            self.meltfrac = npy.delete(self.meltfrac, npy.where(self.id == pid))
            self.phase = npy.delete(self.phase, npy.where(self.id == pid))

        self.id = npy.delete(self.id, npy.where(self.id == pid))

        
    def write(self, fname):
        """
        Write snapshot to file
        """
        io.write_snapshot(self, fname)


    def G2_to_swift(self, mats=[401,400], box=5000.*6.371e8, fname=None, write=False):
        """
        Convert Gadget2 format to Swift format
        """
        self.ensure_matIDs(mats)
        #define U
        if len(self.U) == 0:
            self.U = eos.calcprop('U','rho','S',self.rho,self.S,self.materialIDs)
        #define P
        if len(self.P) == 0:
            self.P = eos.calcprop('P','rho','S',self.rho,self.S,self.materialIDs)
        #set BoxSize
        if npy.ndim(self.header.BoxSize) == 0:
            if self.header.BoxSize < self.x.max():
                self.header.BoxSize = box
        if not fname:
            fname = self.file+'.hdf5'
        self.header.flag_entr_ics = 0
        if write:
            io.write_hdf5(self,fname,units='cgs',mats=mats)


### edit
#    def identify(self, crust=False):
#        self.core = self.iron = npy.where(self.id <= IDOFF, 1, 0)
#        self.mant = self.fors = npy.where(self.id > IDOFF, 1, 0)
### edit end

        
    def summary(self):
        print('N SPH:', self.N)
        print('Total mass:', self.m.sum()/5.972e27, 'Earth masses')
        print('Time:', self.header.time/3600., 'hrs')
        if self.file:
            print('File:', self.file)


    def eq_test(self, threshold=0.01):
        """
        Test for succesful equilibration
        """
        r = npy.sqrt( (self.x-npy.average(self.x,weights=self.m))**2 + (self.y-npy.average(self.y,weights=self.m))**2 + (self.z-npy.average(self.z,weights=self.m))**2 )
        vesc = npy.sqrt( 2*G*self.m.sum()/r.max() )        
        vrms = ( npy.sqrt( ( (self.vx-npy.average(self.vx,weights=self.m))**2 + (self.vy-npy.average(self.vy,weights=self.m))**2 + (self.vz-npy.average(self.vz,weights=self.m))**2 ).mean() ) )
        if vrms <= threshold*vesc:
            return True
        else:
            return False
           

    def bound_mass(self, nrem=1, minbnd=500, maxiter=2000, tol=0.001, reorder=True, discardsmall=False, calc_pot_all=True, save=True):
        """
        Calculate bound remnants
        """
        self.rem = npy.zeros(len(self.id)).astype(int)
    
        for r in range(1,nrem+1):
            prevmass = 0
            icount = 0
            if r==1:
                seed = self.id[self.pot==self.pot.min()][0]
            else:
                if len(self.id[self.rem==0]) < minbnd:
                # if fewer particles than classed as a remnant remain to be tested we are done
                    break
                pid = self.id[self.rem==0]
                px = self.x[self.rem==0]
                py = self.y[self.rem==0]
                pz = self.z[self.rem==0]
                pm = self.m[self.rem==0]
                if calc_pot_all:
                    pot = utils.calc_potential(pm,px,py,pz)
                else:
                    pot = self.pot[self.rem==0]
                seed = pid[(pot==pot.min())][0]

            self.rem[self.id==seed] = r

            while abs((self.m[self.rem==r].sum()-prevmass)/prevmass) > tol and icount < maxiter:
                sm = prevmass = self.m[self.rem==r].astype('float64',copy=False).sum()
                #sx = (self.x[self.rem==r].astype('float64',copy=False)*self.m[self.rem==r].astype('float64',copy=False)).sum()/sm
                #sy = (self.y[self.rem==r].astype('float64',copy=False)*self.m[self.rem==r].astype('float64',copy=False)).sum()/sm
                #sz = (self.z[self.rem==r].astype('float64',copy=False)*self.m[self.rem==r].astype('float64',copy=False)).sum()/sm
                #svx = (self.vx[self.rem==r].astype('float64',copy=False)*self.m[self.rem==r].astype('float64',copy=False)).sum()/sm
                #svy = (self.vy[self.rem==r].astype('float64',copy=False)*self.m[self.rem==r].astype('float64',copy=False)).sum()/sm
                #svz = (self.vz[self.rem==r].astype('float64',copy=False)*self.m[self.rem==r].astype('float64',copy=False)).sum()/sm
                
                sx,sy,sz = utils.com(self.m[self.rem==r].astype('float64',copy=False), self.x[self.rem==r].astype('float64',copy=False), self.y[self.rem==r].astype('float64',copy=False), self.z[self.rem==r].astype('float64',copy=False))
                svx,svy,svz = utils.com_v(self.m[self.rem==r].astype('float64',copy=False), self.vx[self.rem==r].astype('float64',copy=False), self.vy[self.rem==r].astype('float64',copy=False), self.vz[self.rem==r].astype('float64',copy=False))
                
                ke = 0.5 * self.m.astype('float64',copy=False) * ( (self.vx-svx)**2 + (self.vy-svy)**2 + (self.vz-svz)**2 ).astype('float64',copy=False)
                pe = - G * self.m.astype('float64',copy=False) * sm / npy.sqrt( (self.x-sx)**2 + (self.y-sy)**2 + (self.z-sz)**2 ).astype('float64',copy=False)

                self.rem[self.rem==0] = npy.where((ke+pe)[self.rem==0] < 0, r, self.rem[self.rem==0])
            
                icount += 1
        
        if discardsmall:
            for r in range(1,self.rem.max()+1):
                if len(self.m[self.rem==r]) < minbnd:
                    self.rem[self.rem==r] = -r

        if reorder:
            sm = [self.m[self.rem==r].sum() for r in range(1, self.rem.max()+1)]
            order = npy.argsort(sm)
            sbnd = self.rem.copy()
            rr = 1
            for r in order[::-1]:
                self.rem[sbnd==r+1] = rr
                rr+=1
        
        if save:
            io.save_remnant_ids(self)
        
        self.bnd = self.rem

                             
### edit                
    def calc_phase(self,release=False,plot=False):
        """
        Calculate phase and melt/vapor fractions
        """
        ### currently only works for forsterite mantle and iron or alloy core!
        if npy.unique(self.materialIDs)[-1] == 402:
            CoreEOS = eos.select('alloy')
        else:
            CoreEOS = eos.select('iron')
        MantleEOS = eos.select('forsterite')
    
        if release:
            release*=10

        self.vapfrac = npy.zeros(len(self.materialIDs))
        self.meltfrac = npy.zeros(len(self.materialIDs))
        self.phase = npy.zeros(len(self.materialIDs))
    
        FoSsol=scipy.interpolate.interp1d(MantleEOS.mc.Ps*1e10,MantleEOS.mc.Ss*1e3*1e7, bounds_error=False)
        FoSmelt=scipy.interpolate.interp1d(MantleEOS.mc.Pl*1e10,MantleEOS.mc.Sl*1e3*1e7, bounds_error=False)
        CSsol=scipy.interpolate.interp1d(CoreEOS.mc.Ps*1e10,CoreEOS.mc.Ss*1e3*1e7, bounds_error=False)
        CSmelt=scipy.interpolate.interp1d(CoreEOS.mc.Pl*1e10,CoreEOS.mc.Sl*1e3*1e7, bounds_error=False)

        if release:
            self.meltfrac[self.materialIDs >= 300] = (self.S[self.materialIDs >= 300] - FoSsol(release)) / (FoSmelt(release) - FoSsol(release))
            self.meltfrac[self.id < GADGET_EOS_OFFSET] = (self.S[self.id < GADGET_EOS_OFFSET] - CSsol(release)) / (CSmelt(release) - CSsol(release))
        else:
            self.meltfrac[self.materialIDs >= 300] = (self.S[self.materialIDs >= 300] - FoSsol(self.P[self.materialIDs >= 300])) / (FoSmelt(self.P[self.materialIDs >= 300]) - FoSsol(self.P[self.materialIDs >= 300]))
            self.meltfrac[self.id < GADGET_EOS_OFFSET] = (self.S[self.id < GADGET_EOS_OFFSET] - CSsol(self.P[self.id < GADGET_EOS_OFFSET])) / (CSmelt(self.P[self.id < GADGET_EOS_OFFSET]) - CSsol(self.P[self.id < GADGET_EOS_OFFSET]))
        self.meltfrac = npy.where(self.meltfrac < 0, 0., self.meltfrac)
        self.meltfrac = npy.where(self.meltfrac > 1, 1., self.meltfrac)
        self.meltfrac = npy.where(npy.isnan(self.meltfrac), 0., self.meltfrac)
    
        FoSliq=scipy.interpolate.interp1d(MantleEOS.vc.Pl*1e10,MantleEOS.vc.Sl*1e3*1e7, bounds_error=False)
        FoSvap=scipy.interpolate.interp1d(MantleEOS.vc.Pv*1e10,MantleEOS.vc.Sv*1e3*1e7, bounds_error=False)
        CSliq=scipy.interpolate.interp1d(CoreEOS.vc.Pl*1e10,CoreEOS.vc.Sl*1e3*1e7, bounds_error=False)
        CSvap=scipy.interpolate.interp1d(CoreEOS.vc.Pv*1e10,CoreEOS.vc.Sv*1e3*1e7, bounds_error=False)

        if release:
            self.vapfrac[self.materialIDs >= 300] = (self.S[self.materialIDs >= 300] - FoSliq(release)) / (FoSvap(release) - FoSliq(release))
            self.vapfrac[self.id < GADGET_EOS_OFFSET] = (self.S[self.id < GADGET_EOS_OFFSET] - CSliq(release)) / (CSvap(release) - CSliq(release))
        else:
            self.vapfrac[self.materialIDs >= 300] = (self.S[self.materialIDs >= 300] - FoSliq(self.P[self.materialIDs >= 300])) / (FoSvap(self.P[self.materialIDs >= 300]) - FoSliq(self.P[self.materialIDs >= 300]))
            self.vapfrac[self.id < GADGET_EOS_OFFSET] = (self.S[self.id < GADGET_EOS_OFFSET] - CSliq(self.P[self.id < GADGET_EOS_OFFSET])) / (CSvap(self.P[self.id < GADGET_EOS_OFFSET]) - CSliq(self.P[self.id < GADGET_EOS_OFFSET]))
        self.vapfrac = npy.where(self.vapfrac < 0, 0., self.vapfrac)
        self.vapfrac = npy.where(self.vapfrac > 1, 1., self.vapfrac)
        self.vapfrac = npy.where(npy.isnan(self.vapfrac),0.,self.vapfrac)

        self.phase[self.materialIDs >= 300] = npy.where(self.S[self.materialIDs >= 300]<FoSsol(self.P[self.materialIDs>=300]),4,5)
        self.phase[self.id >= GADGET_EOS_OFFSET] = npy.where(self.S > FoSmelt(self.P), 6, self.phase)[self.id >= GADGET_EOS_OFFSET]
        self.phase[self.id >= GADGET_EOS_OFFSET] = npy.where(self.S > FoSliq(self.P), 2, self.phase)[self.id >= GADGET_EOS_OFFSET]
        self.phase[self.id >= GADGET_EOS_OFFSET] = npy.where(self.S > FoSvap(self.P), 7, self.phase)[self.id >= GADGET_EOS_OFFSET]
        # should be P,T!
        self.phase[self.id >= GADGET_EOS_OFFSET] = npy.where((self.P > MantleEOS.cp.P*1e10)*(self.S > MantleEOS.cp.S*1e3*1e7), 8, self.phase)[self.id >= GADGET_EOS_OFFSET]

        self.phase[self.id < GADGET_EOS_OFFSET] = npy.where(self.S < CSsol(self.P), 4, 5)[self.id < GADGET_EOS_OFFSET]
        self.phase[self.id < GADGET_EOS_OFFSET] = npy.where(self.S > CSmelt(self.P), 6, self.phase)[self.id < GADGET_EOS_OFFSET]
        self.phase[self.id < GADGET_EOS_OFFSET] = npy.where(self.S > CSliq(self.P), 2, self.phase)[self.id < GADGET_EOS_OFFSET]
        self.phase[self.id < GADGET_EOS_OFFSET] = npy.where(self.S > CSvap(self.P), 7, self.phase)[self.id < GADGET_EOS_OFFSET]
        # should be P,T!
        self.phase[self.id < GADGET_EOS_OFFSET] = npy.where((self.P > CoreEOS.cp.P*1e10)*(self.S > CoreEOS.cp.S*1e3*1e7), 8, self.phase)[self.id < GADGET_EOS_OFFSET]
### edit end


    def calc_vap_frac(self, plot=False):
        self.calc_phase(plot=False)
