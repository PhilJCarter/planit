### Module for accessing gadget data
###

"""Classes and functions for accessing and manipulating Gadget data."""

import numpy as npy
import matplotlib.pyplot as plt
import struct
import glob
import os
import inspect


IDOFF   = 200000000  # material id offset
BODYOFF = 100000000  # body id offset



class GadgetHeader:
    """Class for Gadget snapshot header.""" 
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
        self.BoxSize = npy.zeros(3)
        self.Omega0 = 0.0
        self.OmegaLambda = 0.0
        self.HubbleParam = 1.0
        self.flag_stellarage = 0
        self.flag_metals = 0
        self.nallhw = npy.zeros(6).astype(int)
        self.flag_entr_ics = ent



class Snapshot:
    """Gadget snapshot class
    
       Includes header and gas particle data, with functions for
       reading and writing snapshots.
       
       load() -- load Gadget snapshot data
       remove() -- remove particle from snapshot
       write() -- save snapshot
       identify() -- determine material types
       summary() -- print basic info
       calc_vap_frac() -- calculate vapour fractions of particles
    """
    def __init__(self):
        self.header = GadgetHeader()
        self.N = 0  # number of SPH particles
        self.pos = npy.zeros(3)  # position vector
        self.vel = npy.zeros(3)  # velocity vector
        self.id = 0  # particle ID
        self.m = 0  # mass
        self.S = 0  # entropy
        self.rho = 0  # density
        self.hsml = 0  # smoothing length
        self.pot = 0  # potential
        #Thermo extension
        self.P = 0  # pressure
        self.T = 0  # temperature
        self.U = 0  # internal energy
        self.cs = 0  # sound speed
        self.accel = 0  # acceleration
        self.dt = 0  # time step
        #Extras
        self.vapfrac = 0  # vapour fraction
        self.file = None
        

    def load(self, fname, thermo=False):
        import h5py
        if not h5py.is_hdf5(fname):
            self.load_G2_1(fname, thermo=thermo)
        else:
            self.load_G2_hdf5(fname, thermo=thermo)
    
    def load_G2_1(self, fname, thermo=False):
        
        f = open(fname, 'rb')

        struct.unpack('i', f.read(4))  #SKIP

        #HEADER
        self.header.npart = npy.array(struct.unpack('iiiiii', f.read(24)))
        self.header.mass = npy.array(struct.unpack('dddddd', f.read(48)))
        (self.header.time, self.header.redshift, self.header.flag_sfr,
          self.header.flag_feedbacktp) = struct.unpack('ddii', f.read(24))
        self.header.npartTotal = npy.array(struct.unpack('iiiiii', f.read(24)))
        (self.header.flag_cooling, self.header.num_files) = struct.unpack('ii', f.read(8))
        self.header.BoxSize = npy.array(struct.unpack('d', f.read(8)))
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
            

        #REARRANGE
        self.pos = npy.array(self.pos).reshape((self.N, 3))
        self.x = self.pos.T[0]
        self.y = self.pos.T[1]
        self.z = self.pos.T[2]

        self.vel = npy.array(self.vel).reshape((self.N, 3))
        self.vx = self.vel.T[0]
        self.vy = self.vel.T[1]
        self.vz = self.vel.T[2]

        if type(self.accel) != int:
            self.accel = npy.array(self.accel).reshape((self.N, 3))
            self.ax = self.accel.T[0]
            self.ay = self.accel.T[1]
            self.az = self.accel.T[2]

        print("Read", self.N, "particles.\n")
        self.file = fname
        f.close()


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


    def write(self, fname):
        import h5py
        if not h5py.is_hdf5(fname):
            self.write_G2_1(self, fname)
        else:
            self.write_G2_hdf5(self, fname)

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
                     self.header.num_files,*self.header.BoxSize,
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



    def identify(self, crust=False):
        self.iron = npy.where(self.id <= IDOFF, 1, 0)
        self.fors = npy.where(self.id > IDOFF, 1, 0)

        
    def summary(self):
        print('N SPH:', self.N)
        print('Total mass:', self.m.sum()/5.972e27, 'Earth masses')
        print('Time:', self.header.time/3600., 'hrs')
        if self.file:
            print('File:', self.file)


    def calc_vap_frac(self,plot=False):
        if __name__ == "__main__":
            dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
        else:
            dir = os.path.dirname(inspect.getfile(Snapshot))
        phaseboundary = dir + '/forsterite_bell.txt'
        phaseboundaryFe = dir + '/iron_bell.txt'
        sd, td = npy.loadtxt(phaseboundary, usecols=(0,1), unpack=True,
                                                                comments='#')
        sdFe, tdFe = npy.loadtxt(phaseboundaryFe, usecols=(0,1), unpack=True,
                                                                comments='#')
        tc = td.max()
        tcFe = tdFe.max()


        liqL = npy.zeros(len(self.T)).astype(int)
        liqH = npy.zeros(len(self.T)).astype(int)
        vapL = npy.zeros(len(self.T)).astype(int)
        vapH = npy.zeros(len(self.T)).astype(int)
        LiqS = npy.zeros(len(self.T)).astype(int)
        VapS = npy.zeros(len(self.T)).astype(int)
        
        for j in range(len(self.T)):
            if self.id[j] <= IDOFF:
                tdome = tdFe
                sdome = sdFe
            else:
                tdome = td
                sdome = sd
            for i in range(1,len(tdome)):
                if tdome[i] > self.T[j] or i == (len(tdome)-1):
                    liqH[j] = i
                    liqL[j] = i-1
                    break
            for i in range(liqH[j], len(tdome)):
                if tdome[i] < self.T[j] or i == (len(tdome)-1):
                    vapH[j] = i-1
                    vapL[j] = i
                    break

            LiqS[j] = sdome[liqL[j]] + ( (sdome[liqH[j]]-sdome[liqL[j]])
                                          / (tdome[liqH[j]]-tdome[liqL[j]])
                                        * (self.T[j]-tdome[liqL[j]]) )
            VapS[j] = sdome[vapL[j]] + ( (sdome[vapH[j]]-sdome[vapL[j]])
                                          / (tdome[vapH[j]]-tdome[vapL[j]])
                                        * (self.T[j]-tdome[vapL[j]]) )

        self.vapfrac = (self.S - LiqS) / (VapS - LiqS)
        self.vapfrac=npy.where(self.S < LiqS, 0., self.vapfrac)
        self.vapfrac=npy.where(self.S > VapS, 1., self.vapfrac)

        self.vapfrac = npy.where(npy.logical_and(self.id <= IDOFF,
                                            self.T > tcFe), -1, self.vapfrac)
        self.vapfrac = npy.where(npy.logical_and(self.id > IDOFF,
                                            self.T > tc), -1, self.vapfrac)


        if plot:
            plt.scatter(self.S, self.T,color='g')
            plt.plot(sd,td,c='b')
            plt.plot(sdFe,tdFe,c='orange')
            for j in range(len(self.T)):
                if self.vapfrac[j] > 0.:
                    plt.scatter(self.S[j],self.T[j],c='r',zorder=2)
            plt.show()




class Impact:
	def __init__(self):
		self.targ = None
		self.proj = None
		self.v = None
		self.b = None
		self.nsnaps = 0
		self.data = None
		
	def load(self,loc,thermo=False):
		flist = sorted(glob.glob(loc+'snapshot_*'))
		Nf1 = [(flist[x].split('/')[-1]).split('_')[1] for x in range(len(flist))]
		Nf = int(sorted(npy.array(Nf1).astype(int))[-1])
		print(Nf)
		self.nsnaps = len(flist)
		if self.nsnaps>2:
			files = npy.arange(0,Nf,1)
		else:
			files = npy.array([0,Nf])
		self.data = npy.ndarray((len(files),),dtype=npy.object)

		for i in range(len(self.data)):
		    self.data[i] = Snapshot()
		    if Nf > 999:
			    self.data[i].load(loc+'snapshot_{:>04d}'.format(int(files[i])))
		    else:
			    self.data[i].load(loc+'snapshot_{:>03d}'.format(int(files[i])))


	def plotseq(self, n=4, type='materials', seq=None, times=None, scale='Mm', potmin=True):
		if not (seq or times):
			if self.nsnaps>n:
				seq = npy.logspace(0,npy.log10(len(imp.data)-1),n).astype(int)
			else:
				seq = npy.arange(self.nsnaps)
		elif seq:
			n = len(seq)
		
		if scale=='Mm':
			scf = 1e8
		elif scale=='km':
			scf = 1e5
		elif scale=='earth' or scale=='Earth':
			scf = 6.371e8
		else:
			scf = scale
		axlim = 4*int(npy.ceil((self.data[0].x.max()-self.data[0].x.min())/scf/8.)) 
		zcut = axlim/200.*scf

		tcut = 500.   #time cut for pre-contact treatment
	
		# number of cells for grid
		Ng = 601j
		Ngz = 15j

		# region to grid/plot
		zmin=-axlim/10.
		zmax=axlim/10.

		zi=npy.linspace(zmin,zmax,int(Ngz.imag))
		X,Y,Z = npy.mgrid[-axlim:axlim:(Ng),-axlim:axlim:(Ng),zmin:zmax:(Ngz)]

		# density limits
		rhomin=5e-5
		rhomax=10.
		# entropy limits
		cmin=1.5
		cmax=10.

		cmap=plt.get_cmap('plasma').copy()
		if type=='density' or type=='rho':        
			cmap.set_under('k')
		else:
			cmap.set_under('w')
	
		fig = plt.figure(figsize=(10,6))
	
		for i in range(len(seq)):
			j = seq[i]
			plt.subplot(1,n,i+1,aspect='equal')
			ti = ( self.data[j].header.time )/3600.
			if self.data[j].header.time != 0 and potmin:
				x = self.data[j].x - (self.data[j].x[self.data[j].pot==self.data[j].pot.min()])[0]
				y = self.data[j].y - (self.data[j].y[self.data[j].pot==self.data[j].pot.min()])[0]
				z = self.data[j].z - (self.data[j].z[self.data[j].pot==self.data[j].pot.min()])[0]
			else:
				x = self.data[j].x
				y = self.data[j].y
				z = self.data[j].z
			modz = npy.abs(z)
			vcut=2*self.data[j].vel.max()

			if type=='materials' or type=='mat':
				plt.scatter(x[modz<zcut]/scf,y[modz<zcut]/scf,s=0.1,c=-(self.data[j].id/BODYOFF).astype(int)[modz<zcut],alpha=1.)
			elif type=='density' or type=='rho':        
				if self.data[j].header.time<tcut:
					vcut=-5.e5 #8
					rhoit = scipy.interpolate.griddata((x[(self.data[j].vx>vcut)*(modz<zcut)]/scf,y[(self.data[j].vx>vcut)*(modz<zcut)]/scf,z[(self.data[j].vx>vcut)*(modz<zcut)]/scf),self.data[j].rho[(self.data[j].vx>vcut)*(modz<zcut)],(X,Y,Z),method='linear',fill_value=1.e-18)
				rhoi = scipy.interpolate.griddata((x[(self.data[j].vx<vcut)*(modz<zcut)]/scf,y[(self.data[j].vx<vcut)*(modz<zcut)]/scf,z[(self.data[j].vx<vcut)*(modz<zcut)]/scf),self.data[j].rho[(self.data[j].vx<vcut)*(modz<zcut)],(X,Y,Z),method='linear',fill_value=1.e-18)
				if self.data[j].header.time<tcut:
					rhoi = npy.where(rhoi>rhoit,rhoi,rhoit)
				if self.data[j].header.time != 0 and potmin:
					coz = (z[self.data[j].pot==self.data[j].pot.min()])[0] #s.z[modz<zcut]
				else:
					coz = 0 #(z[self.data[j].pot==self.data[j].pot.min()])[0] #s.z[modz<zcut]
				nn=(npy.nonzero(zi==(zi[zi<=coz])[-1])[0])[0]
				cols=matplotlib.colors.LogNorm(vmin=rhomin,vmax=rhomax,clip=False)(rhoi[:,:,nn].T)
				cols=cmap(cols)
				ax = plt.gca()
				im = ax.imshow(cols,origin='lower',extent=[-axlim,axlim,-axlim,axlim],cmap=cmap,norm=matplotlib.colors.LogNorm(vmin=rhomin,vmax=rhomax,clip=False))#vmin=rhomin,vmax=rhomax
				ax=plt.gca()
				ax.tick_params(colors='w',which='both',labelcolor='k')
				ax.spines['top'].set_color('w')
				ax.spines['bottom'].set_color('w')
				ax.spines['left'].set_color('w')
				ax.spines['right'].set_color('w')
	
			elif type=='entropy' or type=='ent':        
				if self.data[j].header.time<tcut: #100 #500
					vcut=-5.e5 #8
					vit = scipy.interpolate.griddata((x[(self.data[j].vx>vcut)*(modz<zcut)]/scf,y[(self.data[j].vx>vcut)*(modz<zcut)]/scf,z[(self.data[j].vx>vcut)*(modz<zcut)]/scf),self.data[j].S[(self.data[j].vx>vcut)*(modz<zcut)]/1e7,(X,Y,Z),method='linear',fill_value=1.e-18)
					rhoit = scipy.interpolate.griddata((x[(self.data[j].vx>vcut)*(modz<zcut)]/scf,y[(self.data[j].vx>vcut)*(modz<zcut)]/scf,z[(self.data[j].vx>vcut)*(modz<zcut)]/scf),self.data[j].rho[(self.data[j].vx>vcut)*(modz<zcut)],(X,Y,Z),method='linear',fill_value=1.e-18)
				vi = scipy.interpolate.griddata((x[(self.data[j].vx<vcut)*(modz<zcut)]/scf,y[(self.data[j].vx<vcut)*(modz<zcut)]/scf,z[(self.data[j].vx<vcut)*(modz<zcut)]/scf),self.data[j].S[(self.data[j].vx<vcut)*(modz<zcut)]/1e7,(X,Y,Z),method='linear',fill_value=1.e-18)
				rhoi = scipy.interpolate.griddata((x[(self.data[j].vx<vcut)*(modz<zcut)]/scf,y[(self.data[j].vx<vcut)*(modz<zcut)]/scf,z[(self.data[j].vx<vcut)*(modz<zcut)]/scf),self.data[j].rho[(self.data[j].vx<vcut)*(modz<zcut)],(X,Y,Z),method='linear',fill_value=1.e-18)
				if self.data[j].header.time<tcut: #100 #500
					vi = npy.where(vi>vit,vi,vit)
					rhoi = npy.where(rhoi>rhoit,rhoi,rhoit)
				if self.data[j].header.time != 0 and potmin:
					coz = (z[self.data[j].pot==self.data[j].pot.min()])[0] #s.z[modz<zcut]
				else:
					coz = 0 #(z[self.data[j].pot==self.data[j].pot.min()])[0] #s.z[modz<zcut]
				nn=(npy.nonzero(zi==(zi[zi<=coz])[-1])[0])[0]
				alphas=matplotlib.colors.LogNorm(vmin=0.05*rhomin,vmax=rhomax,clip=True)(rhoi[:,:,nn].T)
				cols = matplotlib.colors.Normalize(vmin=cmin,vmax=cmax,clip=True)(vi[:,:,nn].T)
				cols=cmap(cols)
				cols[..., -1] = alphas    
				ax = plt.gca()
				im = ax.imshow(cols,origin='lower',extent=[-axlim,axlim,-axlim,axlim],vmin=cmin,vmax=cmax,cmap=cmap)

			if i>0:
				plt.gca().set_yticklabels([])
			else:
				if scale=='Mm':
					plt.ylabel('y (Mm)')
				elif scale=='km':
					plt.ylabel('y (km)')
				elif scale=='earth' or scale=='Earth':
					plt.ylabel(r'y (R$_\oplus$)')
				else:
					plt.ylabel('y')
			plt.xlim(-axlim,axlim)
			plt.ylim(-axlim,axlim)
			plt.minorticks_on()    
			if type=='density' or type=='rho':
				plt.text(0.96,0.96,'{:.1f}'.format(ti),ha='right',va='top',fontsize=9,color='w',transform=plt.gca().transAxes)
			else:
				plt.text(0.96,0.96,'{:.1f}'.format(ti),ha='right',va='top',fontsize=9,color='k',transform=plt.gca().transAxes)
			if scale=='Mm':
				plt.xlabel('x (Mm)')
			elif scale=='km':
				plt.xlabel('x (km)')
			elif scale=='earth' or scale=='Earth':
				plt.xlabel(r'x (R$_\oplus$)')
			else:
				plt.xlabel('x')
	
			if type!='materials' and type!='mat':
				if i == len(seq)-1:
	                pbox=plt.gca().get_position()
    	            xw = 0.25
        	        cbar_ax = fig.add_axes([pbox.x1-xw, pbox.y1*1.1, xw, 0.015])
					cbar = fig.colorbar(im,cax=cbar_ax,orientation='horizontal')
					if type=='density' or type=='rho':
						cbar_ax.xaxis.set_label_text(r'$\rho$ (g$\,$cm$^{-3}$)')
					elif type=='entropy' or type=='ent':
						cbar_ax.xaxis.set_label_text(r'$S$ (kJ$\,$K$^{-1}\,$kg$^{-1}$)')
					cbar_ax.xaxis.set_label_position('top')
		plt.subplots_adjust(wspace=0, hspace=0)
		#plt.tight_layout()
		plt.show()




class EOStable:
    """Class for accessing Gadget EoS tables"""    
    def __init__(self):
        self.ND = 0
        self.NS = 0
        self.rho = npy.zeros(self.ND)
        self.S = npy.zeros(self.NS)
        self.P = npy.zeros(self.ND*self.NS)
        self.T = npy.zeros(self.ND*self.NS)
        self.U = npy.zeros(self.ND*self.NS)
        self.cs = npy.zeros(self.ND*self.NS)
        self.hfe = npy.zeros(self.ND*self.NS)
        self.pka = npy.zeros(self.ND*self.NS)
        
    def load(self, fname, flags=False):
        data = ([])
        with open(fname, 'r') as file:
            lines = file.readlines()
            for line in lines:
                data = npy.append(data, line.strip('\n').split(' '))
        data = data[data != ''].astype(float)
        self.ND = data[0].astype(int)
        self.NS = data[1].astype(int)
        self.rho = data[2:2+self.ND]
        self.S = data[2+self.ND : 2+self.ND+self.NS]
        self.P = data[2+self.ND+self.NS : 2+self.ND+self.NS+self.ND*self.NS
                     ].reshape(self.NS,self.ND)
        self.T = data[2+self.ND+self.NS+self.ND*self.NS
                      : 2+self.ND+self.NS+2*self.ND*self.NS
                     ].reshape(self.NS,self.ND)
        self.U = data[2+self.ND+self.NS+2*self.ND*self.NS
                      : 2+self.ND+self.NS+3*self.ND*self.NS
                     ].reshape(self.NS,self.ND)
        self.cs = data[2+self.ND+self.NS+3*self.ND*self.NS
                       : 2+self.ND+self.NS+4*self.ND*self.NS
                      ].reshape(self.NS,self.ND)
        if flags:
            self.hfe = data[2+self.ND+self.NS+4*self.ND*self.NS
                       : 2+self.ND+self.NS+5*self.ND*self.NS
                      ].reshape(self.NS,self.ND)
            self.pka = data[2+self.ND+self.NS+5*self.ND*self.NS
                       : 2+self.ND+self.NS+6*self.ND*self.NS
                      ].reshape(self.NS,self.ND)

    def view(self, q='T', Slow=None, Shigh=None, rholow=None, rhohigh=None):
        if Slow is None:
            Slow = self.S.min()
        if Shigh is None:
            Shigh = self.S.max()
        if rholow is None:
            rholow = self.rho.min()
        if rhohigh is None:
            rhohigh = self.rho.max()
        if q == 'T':
            print('S:', self.S[npy.logical_and(self.S >= Slow,self.S <= Shigh)])
            print('rho:', self.rho[npy.logical_and(self.rho
                                   >= rholow,self.rho<=rhohigh)
                                  ])
            print('T:', (self.T[npy.logical_and(self.S >= Slow,self.S <= Shigh)
                               ])[:, npy.logical_and(self.rho >= rholow,
                                                     self.rho <= rhohigh)
                                 ])
        if q == 'P':
            print('S:', self.S[npy.logical_and(self.S >= Slow,self.S <= Shigh)])
            print('rho:', self.rho[npy.logical_and(self.rho >= rholow,
                                                   self.rho <= rhohigh)
                                  ])
            print('P:', (self.P[npy.logical_and(self.S >= Slow,
                                                self.S<=Shigh)
                               ])[:, npy.logical_and(self.rho >= rholow,
                                                     self.rho <= rhohigh)
                                 ])
        if q == 'U':
            print('S:', self.S[npy.logical_and(self.S >= Slow,self.S <= Shigh)])
            print('rho:', self.rho[npy.logical_and(self.rho >= rholow,
                                                   self.rho <= rhohigh)
                                  ])
            print('U:', (self.U[npy.logical_and(self.S >= Slow,
                                                self.S <= Shigh)
                               ])[:, npy.logical_and(self.rho >= rholow,
                                                     self.rho <= rhohigh)
                                 ])
        if q == 'cs':
            print('S:', self.S[npy.logical_and(self.S >= Slow,self.S <= Shigh)])
            print('rho:', self.rho[npy.logical_and(self.rho >= rholow,
                                                   self.rho<=rhohigh)])
            print('cs:', (self.cs[npy.logical_and(self.S >= Slow,
                                                  self.S <= Shigh)
                                 ])[:, npy.logical_and(self.rho >= rholow,
                                                       self.rho <= rhohigh)
                                   ])


    def write(self, fname):
        NN = self.ND*self.NS
        nd = str(self.ND)
        ns = str(self.NS)
        nn = str(NN)

        f = open(fname, 'wb')
        print(fname,'open')
        f.write(struct.pack('i', int(self.ND)))  #SKIP
        f.write(struct.pack('i', int(self.NS)))  #SKIP

        f.write(struct.pack(nd + 'f', *self.rho))
        f.write(struct.pack(ns + 'f', *self.S))
        f.write(struct.pack(str(NN) + 'f', *self.P.flatten()))
        f.write(struct.pack(str(NN) + 'f', *self.U.flatten()))
        f.write(struct.pack(str(NN) + 'f', *self.T.flatten()))
        f.write(struct.pack(str(NN) + 'f', *self.cs.flatten()))
        f.write(struct.pack(str(NN) + 'f', *self.hfe.flatten()))
        f.write(struct.pack(str(NN) + 'f', *self.pka.flatten()))
        f.close()





if __name__ == "__main__":
    import sys

    file = 'snapshot_000'
    if len(sys.argv) > 1:
        file = sys.argv[1]

    snap0 = Snapshot()
    print(snap0.header.flag_entr_ics)

    snap0.load(file, thermo=False)

    print(snap0.header.flag_entr_ics, snap0.header.flag_metals)
    #print file, snap0.U.max()/1.e11, 0.5*npy.sqrt(snap0.vx**2+snap0.vy**2+snap0.vx**2).max()


    print(snap0.N)
    #print snap0.header.num_files

    #print snap0.id[:5], snap0.id[-5:], snap1.id[:5], snap1.id[-5:]
    #print snap0.m[:5], snap0.m[-5:], snap1.m[:5], snap1.m[-5:]
    #print snap0.x[:5], snap0.x[-5:], snap1.x[:5], snap1.x[-5:]
    #print snap0.vx[:5], snap0.vx[-5:]
    
    
    #snap0.identify()

    #plt.scatter(snap0.x[snap0.fors == 1], snap0.y[snap0.fors == 1], c='b')
    #plt.scatter(snap0.x[snap0.iron == 1], snap0.y[snap0.iron == 1], c='orange')
    #plt.show()
