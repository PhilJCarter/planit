from .eos_table import extEOStable
import numpy as npy

class EOStable(extEOStable):
    def __init__(self):
        extEOStable.__init__(self)
        self.TYPE = ''
        self.womaID = None
        self.NU = None    # only needed for rho-U format tables
        self.U_1D = None  # only needed for rho-U format tables
        #self.name = ''
        
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
        self.T = data[self.ND:].reshape(self.ND,self.NU).T
        