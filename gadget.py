### Module for accessing gadget data
### PJC 12/2017 - CrustStrip / MFIejecta / GIbudgets / SEatmo
###

"""Classes and functions for accessing and manipulating Gadget data."""
import numpy as npy
import matplotlib.pyplot as plt
import struct
import os
import inspect


IDOFF = 200000000  # material id offset
BODYOFF = 100000000  # body id offset



class GadgetHeader:
    """Class for Gadget snapshot header."""	
    def __init__(self, t=0, nfiles=1, ent=1):
        self.npart = npy.zeros(6)
        self.mass = npy.zeros(6)
        self.time = t
        self.redshift = 0
        self.flag_sfr = 0
        self.flagfeedbacktp = 0
        self.npartTotal = npy.zeros(6)
        self.flag_cooling = 0
        self.num_files = nfiles
        self.BoxSize = 0
        self.Omega0 = 0
        self.OmegaLambda = 0
        self.HubbleParam = 1
        self.flag_stellarage = 0
        self.flag_metals = 0
        self.nallhw = npy.zeros(6)
        self.flag_entr_ics = ent



class Snapshot:
    """Gadget snapshot class
    
       Includes header and gas particle data, with functions for
       reading and writing snapshots.
       
       load() -- load Gadget snapshot data
       remove() -- remove particle from snapshot
       write() -- save snapshot
       identify() -- determine material types
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


    def load(self, fname, thermo=False):

        f = open(fname, 'rb')

        struct.unpack('i', f.read(4))  #SKIP

        #HEADER
        self.header.npart = npy.array(struct.unpack('iiiiii', f.read(24)))
        self.header.mass = npy.array(struct.unpack('dddddd', f.read(48)))
        (self.header.time, self.header.redshift, self.header.flag_sfr,
          self.header.flag_feedbacktp) = struct.unpack('ddii', f.read(24))
        self.header.npartTotal = npy.array(struct.unpack('iiiiii', f.read(24)))
        (self.header.flag_cooling, self.header.num_files, self.header.BoxSize,
          self.header.Omega0, self.header.OmegaLambda, self.header.HubbleParam,
          self.header.flag_stellarage,
          self.header.flag_metals) = struct.unpack('iiddddii', f.read(48))
        self.header.nallhw = npy.array(struct.unpack('iiiiii', f.read(24)))
        (self.header.flag_entr_ics,) = struct.unpack('i', f.read(4))
        struct.unpack('60x', f.read(60))

        struct.unpack('i', f.read(4))  #SKIP

        if self.header.num_files != 1:
            print ("WARNING! Number of files:", self.header.num_files,
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

        print "Read", self.N, "particles.\n"
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

        f = open(fname, 'wb')

        f.write(struct.pack('i', self.N))  #SKIP

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

        f.write(struct.pack('i', self.N))  #SKIP

        if self.header.num_files != 1:
            print ("WARNING! Number of files:", self.header.num_files,
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


        print "Wrote", self.N, "particles.\n"
        f.close()


    def identify(self, crust=False):
        self.iron = npy.where(self.id <= IDOFF, 1, 0)
        self.fors = npy.where(self.id > IDOFF, 1, 0)


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
        
    def load(self, fname):
        data = ([])
        with open(fname, 'r') as file:
            for line in file:
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
            print 'S:', self.S[npy.logical_and(self.S >= Slow,self.S <= Shigh)]
            print 'rho:', self.rho[npy.logical_and(self.rho
                                   >= rholow,self.rho<=rhohigh)
                                  ]
            print 'T:', (self.T[npy.logical_and(self.S >= Slow,self.S <= Shigh)
                               ])[:, npy.logical_and(self.rho >= rholow,
                                                     self.rho <= rhohigh)
                                 ]
        if q == 'P':
            print 'S:', self.S[npy.logical_and(self.S >= Slow,self.S <= Shigh)]
            print 'rho:', self.rho[npy.logical_and(self.rho >= rholow,
                                                   self.rho <= rhohigh)
                                  ]
            print 'P:', (self.P[npy.logical_and(self.S >= Slow,
                                                self.S<=Shigh)
                               ])[:, npy.logical_and(self.rho >= rholow,
                                                     self.rho <= rhohigh)
                                 ]
        if q == 'U':
            print 'S:', self.S[npy.logical_and(self.S >= Slow,self.S <= Shigh)]
            print 'rho:', self.rho[npy.logical_and(self.rho >= rholow,
                                                   self.rho <= rhohigh)
                                  ]
            print 'U:', (self.U[npy.logical_and(self.S >= Slow,
                                                self.S <= Shigh)
                               ])[:, npy.logical_and(self.rho >= rholow,
                                                     self.rho <= rhohigh)
                                 ]
        if q == 'cs':
            print 'S:', self.S[npy.logical_and(self.S >= Slow,self.S <= Shigh)]
            print 'rho:', self.rho[npy.logical_and(self.rho >= rholow,
                                                   self.rho<=rhohigh)]
            print 'cs:', (self.cs[npy.logical_and(self.S >= Slow,
                                                  self.S <= Shigh)
                                 ])[:, npy.logical_and(self.rho >= rholow,
                                                       self.rho <= rhohigh)
                                   ]




if __name__ == "__main__":
    import sys

    file = 'snapshot_000'
    if len(sys.argv) > 1:
        file = sys.argv[1]

    snap0 = Snapshot()
    print snap0.header.flag_entr_ics

    snap0.load(file, thermo=False)

    print snap0.header.flag_entr_ics, snap0.header.flag_metals
    #print file, snap0.U.max()/1.e11, 0.5*npy.sqrt(snap0.vx**2+snap0.vy**2+snap0.vx**2).max()


    print snap0.N
    #print snap0.header.num_files

    #print snap0.id[:5], snap0.id[-5:], snap1.id[:5], snap1.id[-5:]
    #print snap0.m[:5], snap0.m[-5:], snap1.m[:5], snap1.m[-5:]
    #print snap0.x[:5], snap0.x[-5:], snap1.x[:5], snap1.x[-5:]
    #print snap0.vx[:5], snap0.vx[-5:]
    
    
    #snap0.identify()

    #plt.scatter(snap0.x[snap0.fors == 1], snap0.y[snap0.fors == 1], c='b')
    #plt.scatter(snap0.x[snap0.iron == 1], snap0.y[snap0.iron == 1], c='orange')
    #plt.show()

    #snap0.load(file+'_iptlecg',thermo=True)
    #snap0.calc_vap_frac()
