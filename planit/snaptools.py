from .main import *
from .utils import *
from . import eos
#from .eos import eosfuncs

import numpy as npy
import scipy
import h5py
import struct
import matplotlib
import matplotlib.pyplot as plt


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
        Automatically attempts read from file if attribute is accessed but not loaded
        """
        if attr in ['x','y','z','vx','vy','vz','pos','vel','id','m','S','rho','hsml','pot','P','T','U','cs'] and self.file:
            if len(super().__getattribute__(attr))==0:
                self.load(self.file, headonly=False, compress=False, thermo=self.inclthermo)
        elif attr in ['rem', 'bnd']:
            if super().__getattribute__(attr) is None:
                if h5py.is_hdf5(self.file):
                    with h5py.File(self.file,'r') as f:
                        if 'RemnantIDs' in f['PartType0'].keys():   
                            self.load(self.file, headonly=False, compress=False, thermo=self.inclthermo)
                        #else:
                        #    #self.bound_mass()
                        #    print('run bound_mass()')
                else:
                    if os.path.exists(self.file+'_rem.txt'):
                        self.load(self.file, headonly=False, compress=False, thermo=self.inclthermo)
                    #else:
                    #    #self.bound_mass()
                    #    print('run bound_mass()')
        elif attr in ['vapfrac','meltfrac','phase']:
            if super().__getattribute__(attr) is None:
                self.calc_phase()
        return super().__getattribute__(attr)

    
    def freedata(self):
        """
        Clear all loaded data
        """
        self.initarrays()

    def ensure_matIDs(self,mats):
        if npy.ndim(self.materialIDs) < 1:
            self.materialIDs = npy.choose( (self.id/GADGET_EOS_OFFSET).astype(int), mats )
                
        
    def load(self, fname, headonly = False, thermo=False, compress=False, mats=[402,400]):
        """
        Loads snapshot data from file
        """
        if not h5py.is_hdf5(fname):
            self.load_G2_1(fname, headonly=headonly, thermo=thermo, compress=compress, mats=mats)
        else:
            self.load_hdf5(fname, headonly=headonly, thermo=thermo, compress=compress)
    
    def load_G2_1(self, fname, headonly=False, thermo=False, compress=False, mats=[402,400]):
        """
        Load a snapshot in Gadget's standard file format (1)
        """
        
        f = open(fname, 'rb')

        struct.unpack('i', f.read(4))  #SKIP

        #HEADER
        self.header.npart = npy.array(struct.unpack('iiiiii', f.read(24)))
        self.header.mass = npy.array(struct.unpack('dddddd', f.read(48)))
        (self.header.time, self.header.redshift, self.header.flag_sfr,
          self.header.flag_feedbacktp) = struct.unpack('ddii', f.read(24))
        self.header.npartTotal = npy.array(struct.unpack('iiiiii', f.read(24)))
        (self.header.flag_cooling, self.header.num_files) = struct.unpack('ii', f.read(8))
        (self.header.BoxSize,) = struct.unpack('d', f.read(8))
        (self.header.Omega0, self.header.OmegaLambda, self.header.HubbleParam,
          self.header.flag_stellarage,
          self.header.flag_metals) = struct.unpack('dddii', f.read(32))
        self.header.nallhw = npy.array(struct.unpack('iiiiii', f.read(24)))
        (self.header.flag_entr_ics,) = struct.unpack('i', f.read(4))
        struct.unpack('60x', f.read(60))

        struct.unpack('i', f.read(4))  #SKIP

        if self.header.num_files != 1:
            print("WARNING! Number of files:", self.header.num_files,
                   ", not currently supported.\n")

        self.N = self.header.npart[0]
        self.file = fname
        self.inclthermo = thermo
        
        if headonly:
            f.close()
            return


        count = str(self.N)     # number of particle values to read
        count3 = str(3*self.N)  # number of values to read for 3-vectors

        #PARTICLE DATA
        struct.unpack('i', f.read(4))  #SKIP
        self.pos = struct.unpack(count3 + 'f', f.read(3*self.N*4))
        struct.unpack('i', f.read(4))  #SKIP

        struct.unpack('i', f.read(4))  #SKIP
        self.vel = struct.unpack(count3 + 'f', f.read(3*self.N*4))
        struct.unpack('i', f.read(4))  #SKIP

        struct.unpack('i', f.read(4))  #SKIP
        self.id = npy.array(struct.unpack(count + 'i', f.read(self.N*4)))
        struct.unpack('i', f.read(4))  #SKIP

        struct.unpack('i', f.read(4))  #SKIP
        self.m = npy.array(struct.unpack(count + 'f', f.read(self.N*4)))
        struct.unpack('i', f.read(4))  #SKIP

        struct.unpack('i', f.read(4))  #SKIP
        self.S = npy.array(struct.unpack(count + 'f', f.read(self.N*4)))
        struct.unpack('i', f.read(4))  #SKIP

        struct.unpack('i', f.read(4))  #SKIP
        self.rho = npy.array(struct.unpack(count + 'f', f.read(self.N*4)))
        struct.unpack('i', f.read(4))  #SKIP

        struct.unpack('i', f.read(4))  #SKIP
        self.hsml = npy.array(struct.unpack(count + 'f', f.read(self.N*4)))
        struct.unpack('i', f.read(4))  #SKIP

        struct.unpack('i', f.read(4))  #SKIP
        self.pot = npy.array(struct.unpack(count + 'f', f.read(self.N*4)))
        struct.unpack('i', f.read(4))  #SKIP

        if thermo:
            if len(f.read(4)) == 4:
                self.P = npy.array(struct.unpack(count + 'f', f.read(self.N*4)))
                struct.unpack('i', f.read(4))  #SKIP
            else:
                self.ensure_matIDs(mats)
                #self.P = npy.zeros(self.N)
                self.P = eos.calcprop('P', 'rho', 'S', self.rho, self.S, self.materialIDs)
            
            if len(f.read(4)) == 4:
                self.T = npy.array(struct.unpack(count + 'f', f.read(self.N*4)))
                struct.unpack('i', f.read(4))  #SKIP
            else:
                self.ensure_matIDs(mats)
                #self.T = npy.zeros(self.N)
                #for i in range(len(self.id)):
                #    self.T[i] = woma.eos.sesame.T_rho_s(self.rho[i]*woma.misc.utils.cgs_to_SI.rho, self.S[i]*woma.misc.utils.cgs_to_SI.s, self.materialIDs[i])
                self.T = eos.calcprop('T', 'rho', 'S', self.rho, self.S, self.materialIDs)
            
            #struct.unpack('i', f.read(4))  #SKIP
            if len(f.read(4)) == 4:
                self.U = npy.array(struct.unpack(count+'f', f.read(self.N*4)))
                struct.unpack('i', f.read(4))  #SKIP
            else:
                self.ensure_matIDs(mats)
                #self.P = npy.zeros(self.N)
                self.U = eos.calcprop('U', 'rho', 'S', self.rho, self.S, self.materialIDs)
            
            #struct.unpack('i', f.read(4))  #SKIP
            if len(f.read(4)) == 4:
                self.cs = npy.array(struct.unpack(count+'f', f.read(self.N*4)))
                struct.unpack('i', f.read(4))  #SKIP
            
#            if len(f.read(4)) == 4: # acceleration near end in _long format
#                self.accel = struct.unpack(count3 + 'f', f.read(3*self.N*4))
#                struct.unpack('i', f.read(4))  #SKIP
            
#            if len(f.read(4)) == 4:
#                self.dt = npy.array(struct.unpack(count+'f', f.read(self.N*4)))
#                struct.unpack('i', f.read(4))  #SKIP
            
        if self.N<=5e9 and compress:
            self.id = self.id.astype('uint32',copy=False)
        if compress:
            self.m = self.m.astype('float32',copy=False)
            self.pos = npy.array(self.pos).astype('float32',copy=False)
            self.vel = npy.array(self.vel).astype('float32',copy=False)
            self.S = self.S.astype('float32',copy=False)
            self.rho = self.rho.astype('float32',copy=False)
            self.hsml = self.hsml.astype('float32',copy=False)
            self.pot = self.pot.astype('float32',copy=False)
            if thermo:
                self.P = self.P.astype('float32',copy=False)
                self.T = self.T.astype('float32',copy=False)
                if len(self.U)>0:
                    self.U = self.U.astype('float32',copy=False)
                if len(self.cs)>0:
                    self.cs = self.cs.astype('float32',copy=False)

        #REARRANGE
        self.pos = npy.array(self.pos).reshape((self.N, 3))
        self.x = self.pos.T[0]
        self.y = self.pos.T[1]
        self.z = self.pos.T[2]

        self.vel = npy.array(self.vel).reshape((self.N, 3))
        self.vx = self.vel.T[0]
        self.vy = self.vel.T[1]
        self.vz = self.vel.T[2]

        #if len(self.accel) > 0:
        #   self.accel = npy.array(self.accel).reshape((self.N, 3))
        #   self.ax = self.accel.T[0]
        #   self.ay = self.accel.T[1]
        #   self.az = self.accel.T[2]

        #print("Read", self.N, "particles.\n")
        f.close()
        
        if os.path.exists(self.file+'_rem.txt'):
            ids,rems = npy.loadtxt(self.file+'_rem.txt', unpack=True)
            if npy.array_equal(ids,self.id):
                self.rem = rems
            else:
                print('array mismatch')

            
    def load_hdf5(self, fname, headonly=False, thermo=False, compress=False, debug=False):
        """
        Load an HDF5 snapshot
        """        
        with h5py.File(fname,'r') as f:
            header= f.get('Header')
            part = f['PartType0']
  #          #pars = f.get('RuntimePars')
            #tid = npy.where(f['PartType0/ParticleIDs'][:] < PROJ_ID_OFFSET)[0]
            #pid = npy.where(f['PartType0/ParticleIDs'][:] >= PROJ_ID_OFFSET)[0]

            if 'Units' in f.keys():
                units = f.get('Units')
                if debug:
                    print('setting conversion factors')
                Lfactor = units.attrs["Unit length in cgs (U_L)"]
                Mfactor = units.attrs["Unit mass in cgs (U_M)"]
                Tfactor = units.attrs["Unit time in cgs (U_t)"]
            else:
                Lfactor = Mfactor = Tfactor = 1.

            self.header.npart = header.attrs['NumPart_ThisFile']
            self.header.mass = header.attrs['MassTable'] * Mfactor
            self.header.time = header.attrs['Time'] * Tfactor
            self.header.redshift = 0.0
            self.header.flag_sfr = 0
            self.header.flag_feedbacktp = 0
            self.header.npartTotal = header.attrs['NumPart_Total']
            self.header.flag_cooling = 0
            self.header.num_files = header.attrs['NumFilesPerSnapshot']
            if npy.ndim(header.attrs['BoxSize'])>0:
                self.header.BoxSize = (header.attrs['BoxSize']).max() * Lfactor
            else:
                self.header.BoxSize = header.attrs['BoxSize'] * Lfactor
            self.header.Omega0 = 0.0
            self.header.OmegaLambda = 0.0
            self.header.HubbleParam = 1.0
            self.header.flag_stellarage = 0
            self.header.flag_metals = 0
            self.header.nallhw = npy.zeros(6).astype(int)
            self.header.flag_entr_ics = header.attrs['Flag_Entropy_ICs']

            self.N = self.header.npart[0]
            self.file = fname
            self.inclthermo = thermo
        
            if headonly:
                f.close()
                return
        
            self.pos = part['Coordinates'][:].reshape((self.header.npart[0], 3)) * Lfactor 
            self.vel = part['Velocities'][:].reshape((self.header.npart[0], 3)) * Lfactor/Tfactor
            if 'MaterialIDs' in part.keys():
                self.materialIDs = part['MaterialIDs'][:]
            # move to conversion routines
            if part['ParticleIDs'][:].max() < GADGET_EOS_OFFSET and len(self.materialIDs)>1:
                self.id = npy.where(self.materialIDs>400,part['ParticleIDs'][:],part['ParticleIDs'][:]+GADGET_EOS_OFFSET)
                self.id = npy.where(self.materialIDs<400,self.id+GADGET_EOS_OFFSET,self.id)
            else:
                self.id = part['ParticleIDs'][:]
            self.m = part['Masses'][:] * Mfactor
            self.rho = part['Densities'][:] * Mfactor/(Lfactor**3)
            self.hsml = part['SmoothingLengths'][:] * Lfactor
            self.U = part['InternalEnergies'][:] * Lfactor**2/(Tfactor**2)
            if 'Entropies' in part.keys() and part['Entropies'][:].max()>0:
                self.S = part['Entropies'][:] * Lfactor**2/(Tfactor**2)
            else:
               ##print('interpolating entropy')
                #self.S = npy.zeros(self.N)
                #for i in range(len(self.id)):
                #    if self.materialIDs[i]>=300:
                #        self.S[i] = woma.eos.sesame.s_u_rho(self.U[i]*woma.misc.utils.cgs_to_SI.u, self.rho[i]*woma.misc.utils.cgs_to_SI.rho, self.materialIDs[i])*woma.misc.utils.cgs_to_SI.inv().s
                self.S = eos.calcprop('S', 'U', 'rho', self.U, self.rho, self.materialIDs)
                ###if i%100000 == 1:
                ### print(self.S[i])
            self.P = part['Pressures'][:] * Mfactor / (Lfactor * Tfactor**2)
            if 'Temperatures' in part.keys():
                self.T = part['Temperatures'][:]
            else:
                #self.T = npy.zeros(self.N)
                #for i in range(len(self.id)):
                #    if self.materialIDs[i]>=300:
                #        self.T[i] = woma.eos.sesame.T_u_rho(self.U[i]*woma.misc.utils.cgs_to_SI.u, self.rho[i]*woma.misc.utils.cgs_to_SI.rho, self.materialIDs[i])#*woma.misc.utils.cgs_to_SI.inv().T
                self.T = eos.calcprop('T', 'U', 'rho', self.U, self.rho, self.materialIDs)
            if 'Potentials' in part.keys():
                self.pot = part['Potentials'][:] * Lfactor**2/(Tfactor**2)
            #else:
            #    self.pot = npy.zeros(self.N) #???            
            if 'RemnantIDs' in part.keys():
                self.rem = part['RemnantIDs'][:]

        if self.N<=5e9 and compress:
            self.id = self.id.astype('uint32',copy=False)
        if compress:
            self.m = self.m.astype('float32',copy=False)
            self.pos = npy.array(self.pos).astype('float32',copy=False)
            self.vel = npy.array(self.vel).astype('float32',copy=False)
            self.S = self.S.astype('float32',copy=False)
            self.rho = self.rho.astype('float32',copy=False)
            self.hsml = self.hsml.astype('float32',copy=False)
            self.pot = self.pot.astype('float32',copy=False)
            if len(self.U)>0:
                    self.U = self.U.astype('float32',copy=False)
            if len(self.P)>0:
                    self.P = self.P.astype('float32',copy=False)
            if thermo:
                if len(self.T)>0:
                    self.T = self.T.astype('float32',copy=False)
                if len(self.cs)>0:
                    self.cs = self.cs.astype('float32',copy=False)

        #REARRANGE
        self.pos = npy.array(self.pos).reshape((self.N, 3))
        self.x = self.pos.T[0]
        self.y = self.pos.T[1]
        self.z = self.pos.T[2]

        self.vel = npy.array(self.vel).reshape((self.N, 3))
        self.vx = self.vel.T[0]
        self.vy = self.vel.T[1]
        self.vz = self.vel.T[2]

        if debug:
            print("Read", self.N, "particles.\n")


    def ic_from_seagen(self, partplanet, thermo=False, init_h=100e5):
        self.load_seagen(partplanet, thermo, init_h)
    
    def load_seagen(self, partplanet, thermo=False, init_h=100e5):
        """
        Assign snapshot particle data from seagen particleplanet
        """
        #HEADER
        self.header.npart = npy.array([partplanet.N_picle, 0, 0, 0, 0, 0])
        self.header.mass = npy.array([0., 0., 0., 0., 0., 0.])
        self.header.time = 0.
        self.header.redshift= 0.
        self.header.flag_sfr = self.header.flag_feedbacktp = self.header.flag_cooling = 0
        self.header.npartTotal = self.header.npart
        self.header.num_files = 1
        self.header.BoxSize = 0.0
        self.header.Omega0 = self.header.OmegaLambda = 0.0
        self.header.HubbleParam = 1.0
        self.header.flag_stellarage = self.header.flag_metals = 0
        self.header.nallhw = npy.array([0, 0, 0, 0, 0, 0])
        self.header.flag_entr_ics = 1

        self.N = self.header.npart[0]

        #PARTICLE DATA
        self.x = partplanet.x
        self.y = partplanet.y
        self.z = partplanet.z
        self.pos = npy.array((self.x, self.y, self.z))
        self.pos = self.pos.T
        
        self.vx = npy.zeros(self.N)
        self.vy = npy.zeros(self.N)
        self.vz = npy.zeros(self.N)
        self.vel = npy.zeros((self.N, 3))

        self.id = npy.where(partplanet.mat==0, npy.arange(len(partplanet.mat)), npy.arange(len(partplanet.mat))-len(partplanet.mat[partplanet.mat==0])+IDOFF )
        self.m = partplanet.m
        self.S = partplanet.S
        self.rho = partplanet.rho
        self.P = partplanet.P
        self.T = partplanet.T
        #self.U = eos.calcprop('U', 'rho', 'S', self.rho, self.S, self.materialIDs)
        
        self.hsml = npy.ones(self.N) * init_h
        self.pot = npy.zeros(self.N)


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
            print("Entropy IC flags must match!")
            
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
        if npy.ndim(body1.materialIDs) > 0:
            self.materialIDs = npy.append(body1.materialIDs,body2.materialIDs)
        if len(body1.U)>0:
            self.U = npy.append(body1.U,body2.U)
        if len(body1.P)>0:
            self.P = npy.append(body1.P,body2.P)
        

    def remove(self, pid):
        """
        Remove particle from Snapshot
        """
        if pid not in self.id:
            return
        self.header.npart[0] = self.header.npart[0]-1
        self.N = self.header.npart[0]

        self.x = npy.delete(self.x, npy.where(self.id == pid))
        self.y = npy.delete(self.y, npy.where(self.id == pid))
        self.z = npy.delete(self.z, npy.where(self.id == pid))
        self.vx = npy.delete(self.vx, npy.where(self.id == pid))
        self.vy = npy.delete(self.vy, npy.where(self.id == pid))
        self.vz = npy.delete(self.vz, npy.where(self.id == pid))
        self.m = npy.delete(self.m, npy.where(self.id == pid))
        self.S = npy.delete(self.S, npy.where(self.id == pid))
        self.rho = npy.delete(self.rho, npy.where(self.id == pid))
        self.hsml = npy.delete(self.hsml, npy.where(self.id == pid))
        self.pot = npy.delete(self.pot, npy.where(self.id == pid))
        #if type(self.P) != int:
        if len(self.P) > 0:
            self.P = npy.delete(self.P, npy.where(self.id == pid))
            self.T = npy.delete(self.T, npy.where(self.id == pid))
            self.U = npy.delete(self.U, npy.where(self.id == pid))
            self.cs = npy.delete(self.cs, npy.where(self.id == pid))
#        if type(self.accel) != int:
#            self.ax = npy.delete(self.ax, npy.where(self.id == pid))
#            self.ay = npy.delete(self.ay, npy.where(self.id == pid))
#            self.az = npy.delete(self.az, npy.where(self.id == pid))
#            self.dt = npy.delete(self.dt, npy.where(self.id == pid))
        #if type(self.vapfrac) != int:
        if self.rem is not None:
            self.rem = npy.delete(self.rem, npy.where(self.id == pid))
        if self.vapfrac is not None:
            self.vapfrac = npy.delete(self.vapfrac, npy.where(self.id == pid))
            self.meltfrac = npy.delete(self.meltfrac, npy.where(self.id == pid))
            self.phase = npy.delete(self.phasefrac, npy.where(self.id == pid))

        self.id = npy.delete(self.id, npy.where(self.id == pid))

        
    def write(self, fname):
        """
        Write snapshot to file
        """
        if not h5py.is_hdf5(fname):
            self.write_G2_1(fname)
        else:
            self.write_hdf5(fname)

    def write_G2_1(self, fname):

        f = open(fname, 'wb')

        f.write(struct.pack('i', 256))  #SKIP

        #HEADER
        f.write(struct.pack('iiiiii', *self.header.npart))
        f.write(struct.pack('dddddd', *self.header.mass))
        f.write(struct.pack('ddii', self.header.time, self.header.redshift,
                     self.header.flag_sfr, self.header.flag_feedbacktp))
        f.write(struct.pack('iiiiii', *self.header.npartTotal))
        f.write(struct.pack('iiddddii', self.header.flag_cooling,
                     self.header.num_files,self.header.BoxSize,
                     self.header.Omega0,self.header.OmegaLambda,
                     self.header.HubbleParam,self.header.flag_stellarage,
                     self.header.flag_metals))
        f.write(struct.pack('iiiiii', *self.header.nallhw))
        f.write(struct.pack('i', self.header.flag_entr_ics))
        f.write(struct.pack('60x'))

        f.write(struct.pack('i', 256))  #SKIP

        if self.header.num_files != 1:
            print("WARNING! Number of files:", self.header.num_files,
                   ", not currently supported.\n")

        count = str(self.N)
        count3 = str(3*self.N)

        #PARTICLE DATA
        f.write(struct.pack('i', self.N))  #SKIP
        f.write(struct.pack(count3 + 'f',
                             *npy.array(self.pos).reshape((3*self.N))))
        f.write(struct.pack('i', self.N))  #SKIP

        f.write(struct.pack('i', self.N))  #SKIP
        f.write(struct.pack(count3 + 'f',
                             *npy.array(self.vel).reshape((3*self.N))))
        f.write(struct.pack('i', self.N))  #SKIP

        f.write(struct.pack('i', self.N))  #SKIP
        f.write(struct.pack(count + 'i', *self.id))
        f.write(struct.pack('i', self.N))  #SKIP

        f.write(struct.pack('i', self.N))  #SKIP
        f.write(struct.pack(count + 'f', *self.m))
        f.write(struct.pack('i', self.N))  #SKIP

        f.write(struct.pack('i', self.N))  #SKIP
        f.write(struct.pack(count + 'f', *self.S))
        f.write(struct.pack('i', self.N))  #SKIP

        f.write(struct.pack('i', self.N))  #SKIP
        f.write(struct.pack(count + 'f', *self.rho))
        f.write(struct.pack('i', self.N))  #SKIP

        f.write(struct.pack('i', self.N))  #SKIP
        f.write(struct.pack(count + 'f', *self.hsml))
        f.write(struct.pack('i', self.N))  #SKIP
        
        f.write(struct.pack('i', self.N))  #SKIP
        f.write(struct.pack(count + 'f', *self.pot))
        f.write(struct.pack('i', self.N))  #SKIP

        print("Wrote", self.N, "particles.\n")
        f.close()

    def write_hdf5(self,outname,units='SI',mats=[401,400]):
    
        self.ensure_matIDs(mats)
                
        if npy.ndim(self.header.flag_entr_ics) < 1:
            #print(self.header.flag_entr_ics)
            if self.header.flag_entr_ics==1:
                print('Entropy in U block')
                intE = self.S
                if units=='SI':
                    intE *= woma.misc.utils.cgs_to_SI.s
            else:   ## normal for converting gadget2-planetary to swift
                #print('Energy in U block')
                intE = self.U
                if units=='SI':
                    intE *= woma.misc.utils.cgs_to_SI.u
        else:
            if self.header.flag_entr_ics.sum()>0:
                print('Entropy in U block. Warning! not expected for Swift!')
                intE = self.S
                if units=='SI':
                    intE *= woma.misc.utils.cgs_to_SI.s
            else:   ## normal for converting gadget2-planetary to swift
                #print('Energy in U block')
                intE = self.U
                if units=='SI':
                    intE *= woma.misc.utils.cgs_to_SI.u
        
        with h5py.File(outname, 'w') as f:
            if units=='SI':
                #replace with generic writing routine
                woma.misc.io.save_particle_data(
                    f,
                    self.pos*woma.misc.utils.cgs_to_SI.l,
                    self.vel*woma.misc.utils.cgs_to_SI.v,
                    self.m*woma.misc.utils.cgs_to_SI.m,
                    self.hsml*woma.misc.utils.cgs_to_SI.l,
                    self.rho*woma.misc.utils.cgs_to_SI.rho,
                    self.P*woma.misc.utils.cgs_to_SI.P,
                    intE,
                    A1_mat_id = self.materialIDs,
                    A1_id=self.id,
                    A1_s=self.S*woma.misc.utils.cgs_to_SI.s,
                    boxsize=self.header.BoxSize*woma.misc.utils.cgs_to_SI.l,
                    #file_to_SI = woma.Conversions(M_earth, R_earth, 1), # mass, length, time
                )
            else:
                #replace with generic writing routine
                woma.misc.io.save_particle_data(
                    f,
                    self.pos,
                    self.vel,
                    self.m,
                    self.hsml,
                    self.rho,
                    self.P,
                    intE,
                    A1_mat_id = self.materialIDs,
                    A1_id=self.id,
                    A1_s=self.S,
                    boxsize=self.header.BoxSize,
                    #file_to_SI = woma.Conversions(M_earth, R_earth, 1), # mass, length, time
                )

            f['Header'].attrs.modify('Time',self.header.time)



    def G2_to_swift(self, mats=[401,400], box=5000.*6.371e8, fname=None):
        self.ensure_matIDs(mats)
        #define U
        if len(self.U) == 0:
        #    self.U = npy.zeros(len(self.id))
        #    for i in range(len(self.id)):
        #        self.U[i] = woma.eos.sesame.Z_rho_Y(self.rho[i]*woma.misc.utils.cgs_to_SI.rho, self.S[i]*woma.misc.utils.cgs_to_SI.s, self.materialIDs[i], 'u', 's')*woma.misc.utils.cgs_to_SI.inv().u
            self.U = eos.calcprop('U','rho','S',self.rho,self.S,self.materialIDs)
        #define P
        if len(self.P) == 0:
        #    self.P = npy.zeros(len(self.id))
        #    for i in range(len(self.id)):
        #        self.P[i] = woma.eos.sesame.Z_rho_Y(self.rho[i]*woma.misc.utils.cgs_to_SI.rho, self.S[i]*woma.misc.utils.cgs_to_SI.s, self.materialIDs[i], 'P', 's')*woma.misc.utils.cgs_to_SI.inv().P
            self.P = eos.calcprop('P','rho','S',self.rho,self.S,self.materialIDs)
        #set BoxSize
        if npy.ndim(self.header.BoxSize) == 0:
            if self.header.BoxSize < self.x.max():
                self.header.BoxSize = box
        if not fname:
            fname = self.file+'.hdf5'
        self.header.flag_entr_ics = 0
        self.write_hdf5(fname,units='SI',mats=mats)


    def identify(self, crust=False):
        self.core = self.iron = npy.where(self.id <= IDOFF, 1, 0)
        self.mant = self.fors = npy.where(self.id > IDOFF, 1, 0)

        
    def summary(self):
        print('N SPH:', self.N)
        print('Total mass:', self.m.sum()/5.972e27, 'Earth masses')
        print('Time:', self.header.time/3600., 'hrs')
        if self.file:
            print('File:', self.file)


    def eq_test(self,threshold=0.01):
        r = npy.sqrt( (self.x-npy.average(self.x,weights=self.m))**2 + (self.y-npy.average(self.y,weights=self.m))**2 + (self.z-npy.average(self.z,weights=self.m))**2 )
        vesc = npy.sqrt( 2*6.67e-8*self.m.sum()/r.max() )        
        vrms = ( npy.sqrt( ( (self.vx-npy.average(self.vx,weights=self.m))**2 + (self.vy-npy.average(self.vy,weights=self.m))**2 + (self.vz-npy.average(self.vz,weights=self.m))**2 ).mean() ) )
        if vrms<=threshold*vesc:
            return True
        else:
            return False
           

    def bound_mass(self, nrem = 1, minbnd = 200, maxiter = 2000, tol = 0.01, reorder=True, discardsmall=False, calc_pot_all=True, save=True):
        #G=6.67e-8
        self.rem = npy.zeros(len(self.id)).astype(int)
    
        for r in range(1,nrem+1):
            prevmass = 1
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
                    pot = calc_potential(pm,px,py,pz)
                else: # and len(self.rem[self.rem==r-1])<minbnd
                    pot = self.pot[self.rem==0]
                seed = pid[(pot==pot.min())][0]

            self.rem[self.id==seed] = r

            while abs((self.m[self.rem==r].sum()-prevmass)/prevmass) > tol and icount < maxiter:
                sm = prevmass = self.m[self.rem==r].astype('float64',copy=False).sum()
                sx = (self.x[self.rem==r].astype('float64',copy=False)*self.m[self.rem==r].astype('float64',copy=False)).sum()/sm
                sy = (self.y[self.rem==r].astype('float64',copy=False)*self.m[self.rem==r].astype('float64',copy=False)).sum()/sm
                sz = (self.z[self.rem==r].astype('float64',copy=False)*self.m[self.rem==r].astype('float64',copy=False)).sum()/sm
                svx = (self.vx[self.rem==r].astype('float64',copy=False)*self.m[self.rem==r].astype('float64',copy=False)).sum()/sm
                svy = (self.vy[self.rem==r].astype('float64',copy=False)*self.m[self.rem==r].astype('float64',copy=False)).sum()/sm
                svz = (self.vz[self.rem==r].astype('float64',copy=False)*self.m[self.rem==r].astype('float64',copy=False)).sum()/sm
            
                ke = 0.5 * self.m.astype('float64',copy=False) * ( (self.vx-svx)**2 + (self.vy-svy)**2 + (self.vz-svz)**2 ).astype('float64',copy=False)
                pe = - G * self.m.astype('float64',copy=False) * sm / npy.sqrt( (self.x-sx)**2 + (self.y-sy)**2 + (self.z-sz)**2 ).astype('float64',copy=False)

                self.rem[self.rem==0] = npy.where((ke+pe)[self.rem==0]<0,r,self.rem[self.rem==0])
            
                icount += 1
            #print(icount)
            #if len(self.id[self.rem==r]) < minbnd and len(self.id[self.rem==r-1]) < minbnd:
            #    break
        
        if discardsmall:
            for r in range(1,self.rem.max()+1):
                if len(self.m[self.rem==r]) < minbnd:
                    self.rem[self.rem==r] = 0

        if reorder:
            sm = [self.m[self.rem==r].sum() for r in range(1,self.rem.max()+1)]
            order = npy.argsort(sm)
            sbnd = self.rem.copy()
            rr = 1
            for r in order[::-1]:
                self.rem[sbnd==r+1] = rr
                rr+=1
        
        if save:
            if h5py.is_hdf5(self.file):
                with h5py.File(self.file,'a') as f:
                    part = f['PartType0']
                    if 'RemnantIDs' in part.keys():
                        part['RemnantIDs'][:] = self.rem
                    else:
                        part.create_dataset('RemnantIDs', data=self.rem, compression='gzip')
            else:
                npy.savetxt(self.file+'_rem.txt', npy.transpose([self.id,self.rem]), header='Id  Remnant', fmt='%d')
        
        self.bnd = self.rem
                
                
                
    def calc_phase(self,release=False,plot=False):
        if npy.unique(self.materialIDs)[-1] == 402:
            CoreEOS = AlloyEOS
        else:
            CoreEOS = IronEOS
        MantleEOS = eos.ANEOSForsterite
    
        if release:
            release*=10

        self.vapfrac = npy.zeros(len(self.materialIDs))
        self.meltfrac = npy.zeros(len(self.materialIDs))
        self.phase = npy.zeros(len(self.materialIDs))
    
        FoSsol=scipy.interpolate.interp1d(ForsteriteEOS.mc.Ps*1e10,ForsteriteEOS.mc.Ss*1e3*1e7,bounds_error=False)
        FoSmelt=scipy.interpolate.interp1d(ForsteriteEOS.mc.Pl*1e10,ForsteriteEOS.mc.Sl*1e3*1e7,bounds_error=False)
        CSsol=scipy.interpolate.interp1d(CoreEOS.mc.Ps*1e10,CoreEOS.mc.Ss*1e3*1e7,bounds_error=False)
        CSmelt=scipy.interpolate.interp1d(CoreEOS.mc.Pl*1e10,CoreEOS.mc.Sl*1e3*1e7,bounds_error=False)

        if release:
            self.meltfrac[self.materialIDs>=300] = (self.S[self.materialIDs>=300] - FoSsol(release)) / (FoSmelt(release) - FoSsol(release))
            self.meltfrac[self.id<GADGET_EOS_OFFSET] = (self.S[self.id<GADGET_EOS_OFFSET] - CSsol(release)) / (CSmelt(release) - CSsol(release))
        else:
            self.meltfrac[self.materialIDs>=300] = (self.S[self.materialIDs>=300] - FoSsol(self.P[self.materialIDs>=300])) / (FoSmelt(self.P[self.materialIDs>=300]) - FoSsol(self.P[self.materialIDs>=300]))
            self.meltfrac[self.id<GADGET_EOS_OFFSET] = (self.S[self.id<GADGET_EOS_OFFSET] - CSsol(self.P[self.id<GADGET_EOS_OFFSET])) / (CSmelt(self.P[self.id<GADGET_EOS_OFFSET]) - CSsol(self.P[self.id<GADGET_EOS_OFFSET]))
        self.meltfrac = npy.where(self.meltfrac < 0, 0., self.meltfrac)
        self.meltfrac = npy.where(self.meltfrac > 1, 1., self.meltfrac)
        self.meltfrac = npy.where(npy.isnan(self.meltfrac),0.,self.meltfrac)
    
        FoSliq=scipy.interpolate.interp1d(ForsteriteEOS.vc.Pl*1e10,ForsteriteEOS.vc.Sl*1e3*1e7,bounds_error=False)
        FoSvap=scipy.interpolate.interp1d(ForsteriteEOS.vc.Pv*1e10,ForsteriteEOS.vc.Sv*1e3*1e7,bounds_error=False)
        CSliq=scipy.interpolate.interp1d(CoreEOS.vc.Pl*1e10,CoreEOS.vc.Sl*1e3*1e7,bounds_error=False)
        CSvap=scipy.interpolate.interp1d(CoreEOS.vc.Pv*1e10,CoreEOS.vc.Sv*1e3*1e7,bounds_error=False)
        #Sliq(self.P)

        if release:
            self.vapfrac[self.materialIDs>=300] = (self.S[self.materialIDs>=300] - FoSliq(release)) / (FoSvap(release) - FoSliq(release))
            self.vapfrac[self.id<GADGET_EOS_OFFSET] = (self.S[self.id<GADGET_EOS_OFFSET] - CSliq(release)) / (CSvap(release) - CSliq(release))
        else:
            self.vapfrac[self.materialIDs>=300] = (self.S[self.materialIDs>=300] - FoSliq(self.P[self.materialIDs>=300])) / (FoSvap(self.P[self.materialIDs>=300]) - FoSliq(self.P[self.materialIDs>=300]))
            self.vapfrac[self.id<GADGET_EOS_OFFSET] = (self.S[self.id<GADGET_EOS_OFFSET] - CSliq(self.P[self.id<GADGET_EOS_OFFSET])) / (CSvap(self.P[self.id<GADGET_EOS_OFFSET]) - CSliq(self.P[self.id<GADGET_EOS_OFFSET]))
        self.vapfrac = npy.where(self.vapfrac < 0, 0., self.vapfrac)
        self.vapfrac = npy.where(self.vapfrac > 1, 1., self.vapfrac)
        self.vapfrac = npy.where(npy.isnan(self.vapfrac),0.,self.vapfrac)

        #if release:
        #    self.phase = npy.where(self.S<FoSsol(release),4,5)
        #else:
        self.phase[self.materialIDs>=300] = npy.where(self.S[self.materialIDs>=300]<FoSsol(self.P[self.materialIDs>=300]),4,5)
        self.phase[self.id>=GADGET_EOS_OFFSET] = npy.where(self.S>FoSmelt(self.P),6,self.phase)[self.id>=GADGET_EOS_OFFSET]
        self.phase[self.id>=GADGET_EOS_OFFSET] = npy.where(self.S>FoSliq(self.P),2,self.phase)[self.id>=GADGET_EOS_OFFSET]
        self.phase[self.id>=GADGET_EOS_OFFSET] = npy.where(self.S>FoSvap(self.P),7,self.phase)[self.id>=GADGET_EOS_OFFSET]
        #should be P,T!
        self.phase[self.id>=GADGET_EOS_OFFSET] = npy.where((self.P>ForsteriteEOS.cp.P*1e10)*(self.S>ForsteriteEOS.cp.S*1e3*1e7),8,self.phase)[self.id>=GADGET_EOS_OFFSET]

        self.phase[self.id<GADGET_EOS_OFFSET] = npy.where(self.S<CSsol(self.P),4,5)[self.id<GADGET_EOS_OFFSET]
        self.phase[self.id<GADGET_EOS_OFFSET] = npy.where(self.S>CSmelt(self.P),6,self.phase)[self.id<GADGET_EOS_OFFSET]
        self.phase[self.id<GADGET_EOS_OFFSET] = npy.where(self.S>CSliq(self.P),2,self.phase)[self.id<GADGET_EOS_OFFSET]
        self.phase[self.id<GADGET_EOS_OFFSET] = npy.where(self.S>CSvap(self.P),7,self.phase)[self.id<GADGET_EOS_OFFSET]
        #should be P,T!
        self.phase[self.id<GADGET_EOS_OFFSET] = npy.where((self.P>CoreEOS.cp.P*1e10)*(self.S>CoreEOS.cp.S*1e3*1e7),8,self.phase)[self.id<GADGET_EOS_OFFSET]


    def calc_vap_frac(self,plot=False):
        self.calc_phase(plot=False)

#         if __name__ == "__main__":
#             dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
#         else:
#             dir = os.path.dirname(inspect.getfile(Snapshot))
#         phaseboundary = dir + '/forsterite_bell.txt'
#         phaseboundaryFe = dir + '/iron_bell.txt'
#         sd, td = npy.loadtxt(phaseboundary, usecols=(0,1), unpack=True,
#                                                                 comments='#')
#         sdFe, tdFe = npy.loadtxt(phaseboundaryFe, usecols=(0,1), unpack=True,
#                                                                 comments='#')
#         tc = td.max()
#         tcFe = tdFe.max()
# 
#         liqL = npy.zeros(len(self.T)).astype(int)
#         liqH = npy.zeros(len(self.T)).astype(int)
#         vapL = npy.zeros(len(self.T)).astype(int)
#         vapH = npy.zeros(len(self.T)).astype(int)
#         LiqS = npy.zeros(len(self.T)).astype(int)
#         VapS = npy.zeros(len(self.T)).astype(int)
#         
#         for j in range(len(self.T)):
#             if self.id[j] <= IDOFF:
#                 tdome = tdFe
#                 sdome = sdFe
#             else:
#                 tdome = td
#                 sdome = sd
#             for i in range(1,len(tdome)):
#                 if tdome[i] > self.T[j] or i == (len(tdome)-1):
#                     liqH[j] = i
#                     liqL[j] = i-1
#                     break
#             for i in range(liqH[j], len(tdome)):
#                 if tdome[i] < self.T[j] or i == (len(tdome)-1):
#                     vapH[j] = i-1
#                     vapL[j] = i
#                     break
# 
#             LiqS[j] = sdome[liqL[j]] + ( (sdome[liqH[j]]-sdome[liqL[j]])
#                                           / (tdome[liqH[j]]-tdome[liqL[j]])
#                                         * (self.T[j]-tdome[liqL[j]]) )
#             VapS[j] = sdome[vapL[j]] + ( (sdome[vapH[j]]-sdome[vapL[j]])
#                                           / (tdome[vapH[j]]-tdome[vapL[j]])
#                                         * (self.T[j]-tdome[vapL[j]]) )
# 
#         self.vapfrac = (self.S - LiqS) / (VapS - LiqS)
#         self.vapfrac=npy.where(self.S < LiqS, 0., self.vapfrac)
#         self.vapfrac=npy.where(self.S > VapS, 1., self.vapfrac)
# 
#         self.vapfrac = npy.where(npy.logical_and(self.id <= IDOFF,
#                                             self.T > tcFe), -1, self.vapfrac)
#         self.vapfrac = npy.where(npy.logical_and(self.id > IDOFF,
#                                             self.T > tc), -1, self.vapfrac)
# 
# 
#         if plot:
#             plt.scatter(self.S, self.T,color='g')
#             plt.plot(sd,td,c='b')
#             plt.plot(sdFe,tdFe,c='orange')
#             for j in range(len(self.T)):
#                 if self.vapfrac[j] > 0.:
#                     plt.scatter(self.S[j],self.T[j],c='r',zorder=2)
#             plt.show()



