from .main import *

class GadgetHeader:
    """Class for Gadget/Swift snapshot header.""" 
    def __init__(self, t=0.0, nfiles=1, ent=1):
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
    """Gadget/Swift snapshot class
    
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
        self.header = GadgetHeader()
        self.N = 0  # number of SPH particles
        self.initarrays()
        #Extras
        self.vapfrac = npy.empty(0) #0  # vapour fraction
        self.file = None
        self.inclthermo = False
        self.materialIDs = None
        self.meltfrac = None
        self.phase = None

    def initarrays(self):
        self.pos = npy.empty(0) #npy.zeros(3)  # position vector
        self.x = npy.empty(0)
        self.y = npy.empty(0)
        self.z = npy.empty(0)
        self.vel = npy.empty(0) #npy.zeros(3)  # velocity vector
        self.vx = npy.empty(0)
        self.vy = npy.empty(0)
        self.vz = npy.empty(0)
        self.id = npy.empty(0) #0  # particle ID
        self.m = npy.empty(0) #0  # mass
        self.S = npy.empty(0) #0  # entropy
        self.rho = npy.empty(0) #0  # density
        self.hsml = npy.empty(0) #0  # smoothing length
        self.pot = npy.empty(0) #0  # potential
        #Thermo extension
        self.P = npy.empty(0) #0  # pressure
        self.T = npy.empty(0) #0  # temperature
        self.U = npy.empty(0) #0  # internal energy
        self.cs = npy.empty(0) #0  # sound speed
        self.accel = npy.empty(0) #0  # acceleration
        self.dt = npy.empty(0) #0  # time step
        

    def __getattribute__(self, attr):
        if attr in ['x','y','z','vx','vy','vz','pos','vel','id','m','S','rho','hsml','pot','P','T','U','cs','accel','dt','vapfrac']:
            if len(super().__getattribute__(attr))==0:
                self.load(self.file, headonly=False, compress=True, thermo=self.inclthermo)
        return super().__getattribute__(attr)
    
    def freedata(self):
        self.initarrays()
        
    def load(self, fname, headonly = False, thermo=False, compress=False):
        if not h5py.is_hdf5(fname):
            self.load_G2_1(fname, headonly=headonly, thermo=thermo, compress=compress)
        else:
            self.load_G2_hdf5(fname, headonly=headonly, thermo=thermo, compress=compress)
    
    def load_G2_1(self, fname, headonly=False, thermo=False, compress=False):
        
        """Load a snapshot in Gadget's standard file format (1)"""
        
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


        count = str(self.N)
        count3 = str(3*self.N)

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
            struct.unpack('i', f.read(4))  #SKIP
            self.P = npy.array(struct.unpack(count + 'f', f.read(self.N*4)))
            struct.unpack('i', f.read(4))  #SKIP
            
            struct.unpack('i', f.read(4))  #SKIP
            self.T = npy.array(struct.unpack(count + 'f', f.read(self.N*4)))
            struct.unpack('i', f.read(4))  #SKIP
            
            #struct.unpack('i', f.read(4))  #SKIP
            if len(f.read(4)) == 4:
                self.U = npy.array(struct.unpack(count+'f', f.read(self.N*4)))
                struct.unpack('i', f.read(4))  #SKIP
            
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

    def load_G2_hdf5(self, fname, headonly=False, thermo=False, compress=False, debug=False):
        
        """Load an HDF5 snapshot"""        
        with h5py.File(fname,'r') as f:
            header= f.get('Header')
            part = f['PartType0']
            pars = f.get('RuntimePars')
            #print(header.attrs.get('NumPart_ThisFile'))
            tid = npy.where(f['PartType0/ParticleIDs'][:] < PROJ_ID_OFFSET)[0]
            pid = npy.where(f['PartType0/ParticleIDs'][:] >= PROJ_ID_OFFSET)[0]
            #print(part['Coordinates'][:])
            #print(part['MaterialIDs'][0])
            #print(part['Masses'][0],units.attrs["Unit mass in cgs (U_M)"])

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
        
            self.pos = part['Coordinates'][:].reshape((self.header.npart[0], 3)) * Lfactor #*E_to_SI.l*utils.SI_to_cgs.l
            self.vel = part['Velocities'][:].reshape((self.header.npart[0], 3)) * Lfactor/Tfactor#*E_to_SI.v*utils.SI_to_cgs.v
            if 'MaterialIDs' in part.keys():
                self.materialIDs = part['MaterialIDs'][:]
            # move to conversion routines
            if part['ParticleIDs'][:].max() < GADGET_EOS_OFFSET and len(self.materialIDs)>1:
                self.id = npy.where(self.materialIDs>400,part['ParticleIDs'][:],part['ParticleIDs'][:]+GADGET_EOS_OFFSET)
            else:
                self.id = part['ParticleIDs'][:]
            self.m = part['Masses'][:] * Mfactor #*E_to_SI.m*utils.SI_to_cgs.m
            self.rho = part['Densities'][:] * Mfactor/(Lfactor**3) #*E_to_SI.rho*utils.SI_to_cgs.rho
            self.hsml = part['SmoothingLengths'][:] * Lfactor #*E_to_SI.l*utils.SI_to_cgs.l
            self.U = part['InternalEnergies'][:] * Lfactor**2/(Tfactor**2) #*E_to_SI.u*utils.SI_to_cgs.u
            #if 'Entropies' not in part.keys():
                #print('interpolating entropy')
            self.S = npy.zeros(self.N)
            for i in range(len(self.id)):
                self.S[i] = woma.eos.sesame.s_u_rho(self.U[i]*woma.misc.utils.cgs_to_SI.u, self.rho[i]*woma.misc.utils.cgs_to_SI.rho, self.materialIDs[i])*woma.misc.utils.cgs_to_SI.inv().s
                ###if i%100000 == 1:
                ### print(self.S[i])
            #else:
            #   self.S = part['Entropies'][:] #*E_to_SI.s*utils.SI_to_cgs.s
            self.P = part['Pressures'][:] * Mfactor / (Lfactor * Tfactor**2) #*E_to_SI.P*utils.SI_to_cgs.P
            if 'Temperatures' in part.keys():
                self.T = part['Temperatures'][:]
            else:
                self.T = npy.zeros(self.N)
                for i in range(len(self.id)):
                    self.T[i] = woma.eos.sesame.T_u_rho(self.U[i]*woma.misc.utils.cgs_to_SI.u, self.rho[i]*woma.misc.utils.cgs_to_SI.rho, self.materialIDs[i])#*woma.misc.utils.cgs_to_SI.inv().T
            if 'Potentials' in part.keys():
                self.pot = part['Potentials'][:] * Lfactor**2/(Tfactor**2)
            else:
                self.pot = npy.zeros(self.N) #???

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

        #if len(self.accel) > 0:
        #   self.accel = npy.array(self.accel).reshape((self.N, 3))
        #   self.ax = self.accel.T[0]
        #   self.ay = self.accel.T[1]
        #   self.az = self.accel.T[2]

        if debug:
            print("Read", self.N, "particles.\n")


    def ic_from_seagen(self, partplanet, thermo=False, init_h=100e5):

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
        self.U = partplanet.u
        
        self.hsml = npy.ones(self.N) * init_h
        self.pot = npy.zeros(self.N)
        


    def remove(self, pid):
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
        if type(self.P) != int:
            self.P = npy.delete(self.P, npy.where(self.id == pid))
            self.T = npy.delete(self.T, npy.where(self.id == pid))
            self.U = npy.delete(self.U, npy.where(self.id == pid))
            self.cs = npy.delete(self.cs, npy.where(self.id == pid))
        if type(self.accel) != int:
            self.ax = npy.delete(self.ax, npy.where(self.id == pid))
            self.ay = npy.delete(self.ay, npy.where(self.id == pid))
            self.az = npy.delete(self.az, npy.where(self.id == pid))
            self.dt = npy.delete(self.dt, npy.where(self.id == pid))
        if type(self.vapfrac) != int:
            self.vapfrac = npy.delete(self.vapfrac, npy.where(self.id == pid))

        self.id = npy.delete(self.id, npy.where(self.id == pid))


    def combine(self, body1, body2, bidoffset=BODYOFF, thermo=False, box=0.0):

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
        
    def write(self, fname):
        if not h5py.is_hdf5(fname):
            self.write_G2_1(fname)
        else:
            self.write_G2_hdf5(fname)

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

    def write_G2_hdf5(self,outname,units='SI',mats=[401,400]):
    
        if npy.ndim(self.materialIDs) < 1:  # needed for WoMa save routine
            materialIDs = npy.choose( (self.id/GADGET_EOS_OFFSET).astype(int), mats )
        
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
                print('Entropy in U block. warning! not expected for swift!')
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

            #print(f['Header'].attrs['Time'])
            f['Header'].attrs.modify('Time',self.header.time)
            #print(f['Header'].attrs['Time'])
            #print(sn.id,f['PartType0']['MaterialIDs'][:])


    def G2_to_swift(self, mats=[401,400], box=5000.*6.371e8, fname=None):
        self.materialIDs = npy.choose( (self.id/GADGET_EOS_OFFSET).astype(int), mats )
        #define U
        if len(self.U) == 0:
            self.U = npy.zeros(len(self.id))
            for i in range(len(self.id)):
                self.U[i] = woma.eos.sesame.Z_rho_Y(self.rho[i]*woma.misc.utils.cgs_to_SI.rho, self.S[i]*woma.misc.utils.cgs_to_SI.s, self.materialIDs[i], 'u', 's')*woma.misc.utils.cgs_to_SI.inv().u
        #define P
        if len(self.P) == 0:
            self.P = npy.zeros(len(self.id))
            for i in range(len(self.id)):
                self.P[i] = woma.eos.sesame.Z_rho_Y(self.rho[i]*woma.misc.utils.cgs_to_SI.rho, self.S[i]*woma.misc.utils.cgs_to_SI.s, self.materialIDs[i], 'P', 's')*woma.misc.utils.cgs_to_SI.inv().P
        #set BoxSize
        if npy.ndim(self.header.BoxSize) == 0:
            if self.header.BoxSize < self.x.max():
                self.header.BoxSize = box
        if not fname:
            fname = self.file+'.hdf5'
        self.header.flag_entr_ics = 0
        self.write_G2_hdf5(fname,units='SI',mats=mats)


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
           

    def bound_mass(self, nrem = 1, minbnd = 200, maxiter = 2000, tol = 0.01, reorder=True, discardsmall=False,calc_pot_all=False):
        G=6.67e-8
        self.bnd = npy.zeros(len(self.id)).astype(int)
    
        for r in range(1,nrem+1):
            prevmass = 1
            icount = 0
            if r==1:
                seed = self.id[self.pot==self.pot.min()][0]
            else:
                if len(self.id[self.bnd==0]) < minbnd:
                # if fewer particles than classed as a remnant remain to be tested we are done
                    break
                pid = self.id[self.bnd==0]
                px = self.x[self.bnd==0]
                py = self.y[self.bnd==0]
                pz = self.z[self.bnd==0]
                pm = self.m[self.bnd==0]
                if not calc_pot_all and len(self.bnd[self.bnd==r-1])<minbnd:
                    pot = self.pot[self.bnd==0]
                else:
                    pot = calc_potential(pm,px,py,pz)
                #pdist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(npy.array([px,py,pz]).T))
                #pot = npy.nansum(npy.where( npy.isinf(-G * pm / pdist),0,-G * pm / pdist ),axis=1)
                seed = pid[(pot==pot.min())][0]
                #seed = self.id[self.pot==self.pot[self.bnd==0].min()][0]
            self.bnd[self.id==seed] = r
            #print(self.pot[self.bnd==0].min())

            #print(abs((self.m[self.bnd==r].sum()-prevmass)/prevmass))
            while abs((self.m[self.bnd==r].sum()-prevmass)/prevmass) > tol and icount < maxiter:
                sm = prevmass = self.m[self.bnd==r].astype('float64',copy=False).sum()
                sx = (self.x[self.bnd==r].astype('float64',copy=False)*self.m[self.bnd==r].astype('float64',copy=False)).sum()/sm
                sy = (self.y[self.bnd==r].astype('float64',copy=False)*self.m[self.bnd==r].astype('float64',copy=False)).sum()/sm
                sz = (self.z[self.bnd==r].astype('float64',copy=False)*self.m[self.bnd==r].astype('float64',copy=False)).sum()/sm
                svx = (self.vx[self.bnd==r].astype('float64',copy=False)*self.m[self.bnd==r].astype('float64',copy=False)).sum()/sm
                svy = (self.vy[self.bnd==r].astype('float64',copy=False)*self.m[self.bnd==r].astype('float64',copy=False)).sum()/sm
                svz = (self.vz[self.bnd==r].astype('float64',copy=False)*self.m[self.bnd==r].astype('float64',copy=False)).sum()/sm
            
                ke = 0.5 * self.m.astype('float64',copy=False) * ( (self.vx-svx)**2 + (self.vy-svy)**2 + (self.vz-svz)**2 ).astype('float64',copy=False)
                pe = - G * self.m.astype('float64',copy=False) * sm / npy.sqrt( (self.x-sx)**2 + (self.y-sy)**2 + (self.z-sz)**2 ).astype('float64',copy=False)
                #print(ke[self.bnd==0],npy.unique(pe[self.bnd==0]),sm,sx,sy,sz,svx,svy,svz)
                #self.bnd = npy.where((ke+pe<0)*(self.bnd==0),r,self.bnd)
                self.bnd[self.bnd==0] = npy.where((ke+pe)[self.bnd==0]<0,r,self.bnd[self.bnd==0])
            
                icount += 1
            #print(icount)
            #if len(self.id[self.bnd==r]) < minbnd and len(self.id[self.bnd==r-1]) < minbnd:
            #    break
        
        if discardsmall:
            for r in range(1,self.bnd.max()+1):
                if len(self.m[self.bnd==r]) < minbnd:
                    self.bnd[self.bnd==r] = 0

        if reorder:
            sm = [self.m[self.bnd==r].sum() for r in range(1,self.bnd.max()+1)]
            order = npy.argsort(sm)
            sbnd = self.bnd.copy()
            rr = 1
            for r in order[::-1]:
                self.bnd[sbnd==r+1] = rr
                rr+=1
                
                
                
    def calc_phase(self,release=False,plot=False):
        if npy.unique(self.materialIDs)[-1] == 402:
            CoreEOS = AlloyEOS
        else:
            CoreEOS = IronEOS
    
        if release:
            release*=10
    
        FoSsol=scipy.interpolate.interp1d(ForsteriteEOS.mc.Ps*1e10,ForsteriteEOS.mc.Ss*1e3*1e7,bounds_error=False)
        FoSmelt=scipy.interpolate.interp1d(ForsteriteEOS.mc.Pl*1e10,ForsteriteEOS.mc.Sl*1e3*1e7,bounds_error=False)
        CSsol=scipy.interpolate.interp1d(CoreEOS.mc.Ps*1e10,CoreEOS.mc.Ss*1e3*1e7,bounds_error=False)
        CSmelt=scipy.interpolate.interp1d(CoreEOS.mc.Pl*1e10,CoreEOS.mc.Sl*1e3*1e7,bounds_error=False)

        if release:
            self.meltfrac = (self.S - FoSsol(release)) / (FoSmelt(release) - FoSsol(release))
            self.meltfrac[self.id<GADGET_EOS_OFFSET] = (self.S[self.id<GADGET_EOS_OFFSET] - CSsol(release)) / (CSmelt(release) - CSsol(release))
        else:
            self.meltfrac = (self.S - FoSsol(self.P)) / (FoSmelt(self.P) - FoSsol(self.P))
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
            self.vapfrac = (self.S - FoSliq(release)) / (FoSvap(release) - FoSliq(release))
            self.vapfrac[self.id<GADGET_EOS_OFFSET] = (self.S[self.id<GADGET_EOS_OFFSET] - CSliq(release)) / (CSvap(release) - CSliq(release))
        else:
            self.vapfrac = (self.S - FoSliq(self.P)) / (FoSvap(self.P) - FoSliq(self.P))
            self.vapfrac[self.id<GADGET_EOS_OFFSET] = (self.S[self.id<GADGET_EOS_OFFSET] - CSliq(self.P[self.id<GADGET_EOS_OFFSET])) / (CSvap(self.P[self.id<GADGET_EOS_OFFSET]) - CSliq(self.P[self.id<GADGET_EOS_OFFSET]))
        self.vapfrac = npy.where(self.vapfrac < 0, 0., self.vapfrac)
        self.vapfrac = npy.where(self.vapfrac > 1, 1., self.vapfrac)
        self.vapfrac = npy.where(npy.isnan(self.vapfrac),0.,self.vapfrac)

        #if release:
        #    self.phase = npy.where(self.S<FoSsol(release),4,5)
        #else:
        self.phase = npy.where(self.S<FoSsol(self.P),4,5)
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



