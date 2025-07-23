#from .main import *
from .snaptools import Snapshot

import numpy as npy
import scipy
import h5py
import glob
import matplotlib
import matplotlib.pyplot as plt


class ImpSnapshot(Snapshot):
    pass


class Impact:
    def __init__(self):
        self.targ = None
        self.proj = None
        self.v = None
        self.b = None
        self.nsnaps = 0
        self.snap = self.data = None
        
    def load(self,loc,thermo=False,inter=1,compress=True,code='swift'):
        Nf2 = 0
        if code=='swift':
            flist = sorted(glob.glob(loc+'snapshot_*.hdf5'))
            flist2 = sorted(glob.glob(loc+'xsnapshot_*.hdf5'))
        else:
            flist = sorted(glob.glob(loc+'snapshot_*'))
            flist2 = sorted(glob.glob(loc+'xsnapshot_*'))
        Nf1 = [(flist[x].split('/')[-1]).split('_')[1].split('.')[0] for x in range(len(flist))]
        Nf2 = [(flist2[x].split('/')[-1]).split('_')[1].split('.')[0] for x in range(len(flist2))]
        if len(Nf1)>0:
            Nf = int(sorted(npy.array(Nf1).astype(int))[-1])
        else:
            Nf = 0
        if len(Nf2)>0:
            Nf2 = int(sorted(npy.array(Nf2).astype(int))[-1])
        else:
            Nf2 = 0
        print(Nf+Nf2)
        self.nsnaps = len(flist)+len(flist2)
        if self.nsnaps>2:
            if Nf2>0:
                files = npy.append(npy.arange(0,Nf2+1,inter),npy.arange(0,Nf+1,inter))
            else:
                files = npy.arange(0,Nf+1,inter)
        elif self.nsnaps>0:
            files = npy.array([0,Nf])
        else:
            files = []
        self.data = npy.ndarray((len(files),),dtype=object)

        for i in range(len(self.data)):
            honly = True
            if i==0 or i==len(self.data)-1:
                honly = False
            self.data[i] = ImpSnapshot()
            #try:
            if code=='swift' or code=='Swift':
                #print(files[i])
                if i<Nf2:
                    self.data[i].load(loc+'xsnapshot_{:>04d}.hdf5'.format(int(files[i])),headonly=honly,thermo=thermo,compress=compress)
                else:
                    self.data[i].load(loc+'snapshot_{:>04d}.hdf5'.format(int(files[i])),headonly=honly,thermo=thermo,compress=compress)
            else:
                self.data[i].load(loc+'snapshot_{:>04d}'.format(int(files[i])),headonly=honly,thermo=thermo,compress=compress)
            #except:
            #    self.data[i].load(loc+'snapshot_{:>03d}'.format(int(files[i])),headonly=honly,thermo=thermo,compress=compress)


    def plotseq(self, n=4, type='materials', seq=None, times=None, scale='Mm', potmin=True, tcut = 3600., zoom=1.):
        """
        tcut -- time cut for pre-contact treatment
        """
        if not (seq or times):
            if self.nsnaps>n:
                seq = npy.logspace(0,npy.log10(len(self.data)-1),n).astype(int)
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

        # region to grid/plot
        zmin=-axlim/10.
        zmax=axlim/10.

        axlim /= zoom
    
        # number of cells for grid
        Ng = 601j
        Ngz = 21j


        zi=npy.linspace(zmin,zmax,int(Ngz.imag))
        X,Y,Z = npy.mgrid[-axlim:axlim:(Ng),-axlim:axlim:(Ng),zmin:zmax:(Ngz)]

        # density limits
        rhomin=5e-6
        rhomax=10.
        # entropy limits
        cmin=1.5
        cmax=10.    
        #phase flag limits
        phmin=2.5
        phmax=8.5

        cmap=plt.get_cmap('plasma')#.copy()
        cmapphase = plt.get_cmap('plasma', 6)#.copy()
        cmapphase=matplotlib.colors.ListedColormap(cmapphase.colors[[0,1,2,3,4,5],:])
        cmapphase.set_under('w')
        if type=='density' or type=='rho':        
            cmap.set_under('k')
        else:
            cmap.set_under('w')
    
        fig = plt.figure(figsize=(10,6))
    
        for i in range(len(seq)):
            j = seq[i]
            plt.subplot(1,n,i+1,aspect='equal')
            ti = ( self.data[j].header.time )/3600.
            if npy.ndim(ti)>0:
                ti=ti[0]
            if potmin:  #self.data[j].header.time >tcut and
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
                im = plt.scatter(x[modz<zcut]/scf,y[modz<zcut]/scf,s=0.1,c=-(self.data[j].id/PROJ_ID_OFFSET).astype(int)[modz<zcut],alpha=1.)
            elif type=='density' or type=='rho':        
                if self.data[j].header.time<tcut:
                    vcut=0.5*self.data[0].vel.min() #-5.e5 #8
                    rhoit = scipy.interpolate.griddata((x[(self.data[j].vx>vcut)*(modz<zcut)]/scf,y[(self.data[j].vx>vcut)*(modz<zcut)]/scf,z[(self.data[j].vx>vcut)*(modz<zcut)]/scf),self.data[j].rho[(self.data[j].vx>vcut)*(modz<zcut)],(X,Y,Z),method='linear',fill_value=rhomin/1000.)
                rhoi = scipy.interpolate.griddata((x[(self.data[j].vx<vcut)*(modz<zcut)]/scf,y[(self.data[j].vx<vcut)*(modz<zcut)]/scf,z[(self.data[j].vx<vcut)*(modz<zcut)]/scf),self.data[j].rho[(self.data[j].vx<vcut)*(modz<zcut)],(X,Y,Z),method='linear',fill_value=rhomin/1000.)
                if self.data[j].header.time<tcut:
                    rhoi = npy.where(rhoi>rhoit,rhoi,rhoit)
                if self.data[j].header.time != 0 and potmin:
                    coz = (z[self.data[j].pot==self.data[j].pot.min()])[0] #s.z[modz<zcut]
                else:
                    coz = 0 #(z[self.data[j].pot==self.data[j].pot.min()])[0] #s.z[modz<zcut]
                nn=(npy.nonzero(zi==(zi[zi<=coz])[-1])[0])[0]
                #cols=matplotlib.colors.LogNorm(vmin=rhomin,vmax=rhomax,clip=False)(rhoi[:,:,nn].T)
                #cols=cmap(cols)
                ax = plt.gca()
                im = ax.imshow(rhoi[:,:,nn].T,origin='lower',extent=[-axlim,axlim,-axlim,axlim],cmap=cmap,norm=matplotlib.colors.LogNorm(vmin=rhomin,vmax=rhomax,clip=False))
                ax=plt.gca()
                ax.tick_params(colors='w',which='both',labelcolor='k')
                ax.spines['top'].set_color('w')
                ax.spines['bottom'].set_color('w')
                ax.spines['left'].set_color('w')
                ax.spines['right'].set_color('w')
    
            elif type in ['entropy','ent','S']:        
                if self.data[j].header.time<tcut:
                    vcut=0.5*self.data[0].vel.min() #-5.e5
                    vit = scipy.interpolate.griddata((x[(self.data[j].vx>vcut)*(modz<zcut)]/scf,y[(self.data[j].vx>vcut)*(modz<zcut)]/scf,z[(self.data[j].vx>vcut)*(modz<zcut)]/scf),self.data[j].S[(self.data[j].vx>vcut)*(modz<zcut)]/1e7,(X,Y,Z),method='linear',fill_value=1.e-18)
                    rhoit = scipy.interpolate.griddata((x[(self.data[j].vx>vcut)*(modz<zcut)]/scf,y[(self.data[j].vx>vcut)*(modz<zcut)]/scf,z[(self.data[j].vx>vcut)*(modz<zcut)]/scf),self.data[j].rho[(self.data[j].vx>vcut)*(modz<zcut)],(X,Y,Z),method='linear',fill_value=1.e-18)
                vi = scipy.interpolate.griddata((x[(self.data[j].vx<vcut)*(modz<zcut)]/scf,y[(self.data[j].vx<vcut)*(modz<zcut)]/scf,z[(self.data[j].vx<vcut)*(modz<zcut)]/scf),self.data[j].S[(self.data[j].vx<vcut)*(modz<zcut)]/1e7,(X,Y,Z),method='linear',fill_value=1.e-18)
                rhoi = scipy.interpolate.griddata((x[(self.data[j].vx<vcut)*(modz<zcut)]/scf,y[(self.data[j].vx<vcut)*(modz<zcut)]/scf,z[(self.data[j].vx<vcut)*(modz<zcut)]/scf),self.data[j].rho[(self.data[j].vx<vcut)*(modz<zcut)],(X,Y,Z),method='linear',fill_value=1.e-18)
                if self.data[j].header.time<tcut:
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
            elif type=='phase':
                self.data[j].calc_phase()
                phase = npy.where(self.data[j].phase<=6,self.data[j].phase-1,self.data[j].phase)
                phase = npy.where(phase<2,6,phase)
                im = plt.scatter(x[modz<zcut]/scf,y[modz<zcut]/scf,s=0.1,c=phase[modz<zcut],alpha=1.,cmap=cmapphase,norm=matplotlib.colors.Normalize(vmin=phmin,vmax=phmax,clip=False),rasterized=True)
    #s=0.8

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
                    elif type in ['entropy','ent','S']:
                        cbar_ax.xaxis.set_label_text(r'$S$ (kJ$\,$K$^{-1}\,$kg$^{-1}$)')
                    elif type=='phase':
                        cbar.ax.set_xticklabels(['','s','s+l','l','l+v','v','scf'])  #  colorbar ['s','s+l','l','l+v']
                        cbar_ax.xaxis.set_label_text(r'Phase')
                    cbar_ax.xaxis.set_label_position('top')
        plt.subplots_adjust(wspace=0, hspace=0)
        #plt.tight_layout()
        plt.show()
