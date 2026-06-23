"""
   planit impact class and analysis tools
"""

from .snaptools import Snapshot
from .utils import *

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
        
    def load(self,loc,thermo=False,inter=1,compress=True,code='swift',ndigits=4,prefix='snapshot',prefix2=None):
        Nf2 = 0
        if code=='swift':
            flist = sorted(glob.glob(loc+prefix+'_*.hdf5'))
            if prefix2:
                flist2 = sorted(glob.glob(loc+prefix2+'_*.hdf5'))
        else:
            flist = sorted(glob.glob(loc+prefix+'_*'))
            if prefix2:
                flist2 = sorted(glob.glob(loc+prefix2+'_*'))
        Nf1 = [(flist[x].split('/')[-1]).split('_')[1].split('.')[0] for x in range(len(flist))]
        if prefix2:
            Nf2 = [(flist2[x].split('/')[-1]).split('_')[1].split('.')[0] for x in range(len(flist2))]
        if len(Nf1)>0:
            Nf = int(sorted(npy.array(Nf1).astype(int))[-1])
        else:
            Nf = 0
        if prefix2 and len(Nf2)>0:
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
            if code=='swift' or code=='Swift':
                if i<Nf2:
                    self.data[i].load(loc+prefix2+'_{:>0{width}}d}.hdf5'.format(int(files[i]),width=ndigits),headonly=honly,thermo=thermo,compress=compress)
                else:
                    self.data[i].load(loc+prefix+'_{:>0{width}d}.hdf5'.format(int(files[i]),width=ndigits),headonly=honly,thermo=thermo,compress=compress)
            else:
                self.data[i].load(loc+prefix+'_{:>0{width}d}'.format(int(files[i]),width=ndigits),headonly=honly,thermo=thermo,compress=compress)


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
                # reorder phases for plotting
                phase = npy.where(self.data[j].phase<=6,self.data[j].phase-1,self.data[j].phase)
                phase = npy.where(phase<2,6,phase)
                im = plt.scatter(x[modz<zcut]/scf,y[modz<zcut]/scf,s=0.1,c=phase[modz<zcut],alpha=1.,cmap=cmapphase,norm=matplotlib.colors.Normalize(vmin=phmin,vmax=phmax,clip=False),rasterized=True) #s=0.8

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
                        #cbar.ax.set_xticklabels(['','s','s+l','l','l+v','v','scf'])  #  colorbar ['s','s+l','l','l+v']
                        cbar.ax.set_xticklabels(['s','s+l','l','l+v','v','scf'])  #  colorbar ['s','s+l','l','l+v']
                        cbar_ax.xaxis.set_label_text(r'Phase')
                    cbar_ax.xaxis.set_label_position('top')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()



def multiplotseq(imps, n=4, types='materials', seqs=None, times=None, scale='Mm', potmin=True, zoom=1.):
    if (not isinstance(imps,list)) and (not isinstance(types,list)):
        return imps.plotseq(n=n,type=types,seq=seqs,times=times,scale=scale,potmin=potmin,zoom=zoom)
    elif not isinstance(imps,list):
        imps=[imps,]
    elif not isinstance(types,list):
        types=[types,]
    if not (seqs or times):
        if imps[0].nsnaps>n:
            seqs = npy.logspace(0,npy.log10(len(imps[0].data)-1),n).astype(int)
        else:
            seqs = npy.arange(imps[0].nsnaps)
    elif not isinstance(seqs,list):
        n = len(seqs)
        seqs = [seqs,]
    elif seqs:
        n = [len(s) for s in seqs]
        if not all(x==n[0] for x in n):
            print('Warning: unequal sequences will produce ugly plots')
    
    if not isinstance(zoom,list):
        zoom=[zoom,]
        
    if scale=='Mm':
        scf = 1e8
    elif scale=='km':
        scf = 1e5
    elif scale=='earth' or scale=='Earth':
        scf = 6.371e8
    else:
        scf = scale
    
    tcut = 3600.   #time cut for pre-contact treatment
    
    # number of cells for grid
    Ng = 2401j
    Ngz = 21j

    scsize = 1.5

    # density limits
    rhomin=1e-5
    rhomax=10.
    # entropy limits
    cmin=1.5
    cmax=10.
    # pressure limits
    Pmin=1.e-9
    Pmax=1000.
    #phase flag limits
    phmin=2.5
    phmax=8.5
    
    m = 1
    k = 0
    leftcol = None
    toprow = None
    
    fig = plt.figure()#figsize=(10,4))
    fig.set_figwidth(8.)
    fig.set_figheight(8./max(n)*(len(imps)*len(types)) + 10.)#-0.1
    gs = fig.add_gridspec(ncols=max(n), nrows=len(imps)*len(types),wspace=0,hspace=0,right=0.98,top=0.96)
    print(max(n),len(imps)*len(types))
    
    for imp, seq, zoomfac in zip(imps, seqs, zoom):#, types):
        axlim = 2*int(npy.ceil((imp.data[0].x.max()-imp.data[0].x.min())/scf/8.)) #4
        zcut = axlim/500.*scf #200
        # region to grid/plot
        zmin=-axlim/10.
        zmax=axlim/10.
        axlim /= zoomfac
    
        zlim = zmax/5.*scf

        zi=npy.linspace(zmin,zmax,int(Ngz.imag))
        X,Y,Z = npy.mgrid[-axlim:axlim:(Ng),-axlim:axlim:(Ng),zmin:zmax:(Ngz)]
        
        l = 0
        
        for type in types:
            n = len(seq)
            #if type in ['density','rho','pressure','P',]:
            if l==0:
                cmap = plt.get_cmap('cmr.bubblegum') #'plasma'
                cmapphase = plt.get_cmap('plasma', 6)
                #cmapphase = matplotlib.colors.ListedColormap(plt.get_cmap('magma', 7).colors[1:])#.copy()
            elif l==1 and len(types)>2:
                cmap = plt.get_cmap('cmr.bubblegum')#cmocean.cm.haline #plt.get_cmap('magma')#.reversed()
                cmapphase = plt.get_cmap('cmr.bubblegum', 6)
            elif l==2 or l==1:
                cmap = plt.get_cmap('cmr.eclipse')
                cmapphase = plt.get_cmap('cmr.eclipse', 6)
            else:
                cmap=plt.get_cmap('plasma')#.copy()
                cmapphase = plt.get_cmap('plasma', 6)#.copy()
            cmapphase=matplotlib.colors.ListedColormap(cmapphase.colors[[0,1,2,3,4,5],:])
            cmapphase.set_under('w')
            if type=='density' or type=='rho':        
                cmap.set_under('k')
            else:
                cmap.set_under('w')

            for i in range(len(seq)):
                j = seq[i]
                #plt.subplot(len(imps),n,(m-1)*n + i+1,aspect='equal')
                fig.add_subplot(gs[m-1,i],aspect='equal')
                ti = ( imp.data[j].header.time )/3600.
                if npy.ndim(ti)>0:
                    ti=ti[0]
                if imp.data[j].header.time > tcut and potmin:
                    x = imp.data[j].x - (imp.data[j].x[imp.data[j].pot==imp.data[j].pot.min()])[0]
                    y = imp.data[j].y - (imp.data[j].y[imp.data[j].pot==imp.data[j].pot.min()])[0]
                    z = imp.data[j].z - (imp.data[j].z[imp.data[j].pot==imp.data[j].pot.min()])[0]
                else:
                    x = imp.data[j].x
                    y = imp.data[j].y
                    z = imp.data[j].z
                modz = npy.abs(z)
                vcut=2*imp.data[j].vel.max()

                if type=='materials' or type=='mat':
                    select = (modz<zcut)*(x<=axlim*scf)*(x>=-axlim*scf)*(y<=axlim*scf)*(y>=-axlim*scf)
                    plt.scatter(x[select]/scf,y[select]/scf,s=scsize,c=(imp.data[j].id/BODYOFF).astype(int)[select],alpha=1.,cmap=cmap.reversed(),vmax=(imp.data[0].id/BODYOFF).astype(int).max(),vmin=0,ec=None,rasterized=True)
                    #print(npy.unique(-(imp.data[j].id/BODYOFF).astype(int)[modz<zcut]))
                elif type=='density' or type=='rho':        
                    if imp.data[j].header.time<tcut:
                        vcut = 0.5*imp.data[0].vel.min() #-5.e5 #8
                        rhoit = scipy.interpolate.griddata((x[(imp.data[j].vx>vcut)*(modz<zlim)]/scf,y[(imp.data[j].vx>vcut)*(modz<zlim)]/scf,z[(imp.data[j].vx>vcut)*(modz<zlim)]/scf),imp.data[j].rho[(imp.data[j].vx>vcut)*(modz<zlim)],(X,Y,Z),method='linear',fill_value=1.e-18)
                    rhoi = scipy.interpolate.griddata((x[(imp.data[j].vx<vcut)*(modz<zlim)]/scf,y[(imp.data[j].vx<vcut)*(modz<zlim)]/scf,z[(imp.data[j].vx<vcut)*(modz<zlim)]/scf),imp.data[j].rho[(imp.data[j].vx<vcut)*(modz<zlim)],(X,Y,Z),method='linear',fill_value=1.e-18)
                    if imp.data[j].header.time<tcut:
                        rhoi = npy.where(rhoi>rhoit,rhoi,rhoit)
                    if imp.data[j].header.time != 0 and potmin:
                        coz = (z[imp.data[j].pot==imp.data[j].pot.min()])[0]/scf #s.z[modz<zcut]
                    else:
                        coz = 0 #(z[imp.data[j].pot==imp.data[j].pot.min()])[0] #s.z[modz<zcut]
                    nn=(npy.nonzero(zi==(zi[zi<=coz])[-1])[0])[0]
                    cols = rhoi[:,:,nn].T
                    #cols=matplotlib.colors.LogNorm(vmin=rhomin,vmax=rhomax,clip=False)(rhoi[:,:,nn].T)
                    #cols=cmap(cols)
                    ax = plt.gca()
                    im = ax.imshow(cols,origin='lower',extent=[-axlim,axlim,-axlim,axlim],cmap=cmap,vmin=rhomin,vmax=rhomax,rasterized=True)#,norm=matplotlib.colors.LogNorm(vmin=rhomin,vmax=rhomax,clip=False))#vmin=rhomin,vmax=rhomax
                    ax.tick_params(colors='w',which='both',labelcolor='k')
                    ax.spines['top'].set_color('w')
                    ax.spines['bottom'].set_color('w')
                    ax.spines['left'].set_color('w')
                    ax.spines['right'].set_color('w')

                elif type=='pressure' or type=='P':        
                    if imp.data[j].header.time<tcut:
                        vcut = 0.5*imp.data[0].vel.min() #-5.e5 #8
                        Pit = scipy.interpolate.griddata((x[(imp.data[j].vx>vcut)*(modz<zlim)]/scf,y[(imp.data[j].vx>vcut)*(modz<zlim)]/scf,z[(imp.data[j].vx>vcut)*(modz<zlim)]/scf),imp.data[j].P[(imp.data[j].vx>vcut)*(modz<zlim)]/1e9,(X,Y,Z),method='linear',fill_value=1.e-18)
                    Pi = scipy.interpolate.griddata((x[(imp.data[j].vx<vcut)*(modz<zlim)]/scf,y[(imp.data[j].vx<vcut)*(modz<zlim)]/scf,z[(imp.data[j].vx<vcut)*(modz<zlim)]/scf),imp.data[j].P[(imp.data[j].vx<vcut)*(modz<zlim)]/1e9,(X,Y,Z),method='linear',fill_value=1.e-18)
                    if imp.data[j].header.time<tcut:
                        Pi = npy.where(Pi>Pit,Pi,Pit)
                    if imp.data[j].header.time != 0 and potmin:
                        coz = (z[imp.data[j].pot==imp.data[j].pot.min()])[0]/scf #s.z[modz<zcut]
                    else:
                        coz = 0 #(z[imp.data[j].pot==imp.data[j].pot.min()])[0] #s.z[modz<zcut]
                    nn=(npy.nonzero(zi==(zi[zi<=coz])[-1])[0])[0]
                    cols=matplotlib.colors.LogNorm(vmin=Pmin,vmax=Pmax,clip=False)(Pi[:,:,nn].T)
                    cols=cmap(cols)
                    ax = plt.gca()
                    im = ax.imshow(cols,origin='lower',extent=[-axlim,axlim,-axlim,axlim],cmap=cmap,vmin=Pmin,vmax=Pmax,norm=matplotlib.colors.LogNorm(vmin=Pmin,vmax=Pmax,clip=False))#vmin=rhomin,vmax=rhomax
                    ax.tick_params(colors='k',which='both',labelcolor='k')
                    ax.spines['top'].set_color('k')
                    ax.spines['bottom'].set_color('k')
                    ax.spines['left'].set_color('k')
                    ax.spines['right'].set_color('k')

                elif type in ['entropy','ent','S']:        
                    if imp.data[j].header.time<tcut: #100 #500
                        vcut = 0.5*imp.data[0].vel.min() #-5.e5 #8
                        vit = scipy.interpolate.griddata((x[(imp.data[j].vx>vcut)*(modz<zlim)]/scf,y[(imp.data[j].vx>vcut)*(modz<zlim)]/scf,z[(imp.data[j].vx>vcut)*(modz<zlim)]/scf),imp.data[j].S[(imp.data[j].vx>vcut)*(modz<zlim)]/1e7,(X,Y,Z),method='linear',fill_value=1.e-18)
                        rhoit = scipy.interpolate.griddata((x[(imp.data[j].vx>vcut)*(modz<zlim)]/scf,y[(imp.data[j].vx>vcut)*(modz<zlim)]/scf,z[(imp.data[j].vx>vcut)*(modz<zlim)]/scf),imp.data[j].rho[(imp.data[j].vx>vcut)*(modz<zlim)],(X,Y,Z),method='linear',fill_value=1.e-18)
                    vi = scipy.interpolate.griddata((x[(imp.data[j].vx<vcut)*(modz<zlim)]/scf,y[(imp.data[j].vx<vcut)*(modz<zlim)]/scf,z[(imp.data[j].vx<vcut)*(modz<zlim)]/scf),imp.data[j].S[(imp.data[j].vx<vcut)*(modz<zlim)]/1e7,(X,Y,Z),method='linear',fill_value=1.e-18)
                    rhoi = scipy.interpolate.griddata((x[(imp.data[j].vx<vcut)*(modz<zlim)]/scf,y[(imp.data[j].vx<vcut)*(modz<zlim)]/scf,z[(imp.data[j].vx<vcut)*(modz<zlim)]/scf),imp.data[j].rho[(imp.data[j].vx<vcut)*(modz<zlim)],(X,Y,Z),method='linear',fill_value=1.e-18)
                    if imp.data[j].header.time<tcut: #100 #500
                        vi = npy.where(vi>vit,vi,vit)
                        rhoi = npy.where(rhoi>rhoit,rhoi,rhoit)
                    if imp.data[j].header.time != 0 and potmin:
                        coz = (z[imp.data[j].pot==imp.data[j].pot.min()])[0]/scf #s.z[modz<zcut]
                    else:
                        coz = 0 #(z[imp.data[j].pot==imp.data[j].pot.min()])[0] #s.z[modz<zcut]
                    nn=(npy.nonzero(zi==(zi[zi<=coz])[-1])[0])[0]
                    alphas=matplotlib.colors.LogNorm(vmin=0.05*rhomin,vmax=rhomax,clip=True)(rhoi[:,:,nn].T)
                    cols = matplotlib.colors.Normalize(vmin=cmin,vmax=cmax,clip=True)(vi[:,:,nn].T)
                    cols=cmap(cols)
                    cols[..., -1] = alphas    
                    ax = plt.gca()
                    im = ax.imshow(cols,origin='lower',extent=[-axlim,axlim,-axlim,axlim],vmin=cmin,vmax=cmax,cmap=cmap)
                elif type=='phase':
                    select = (modz<zcut)*(x<=axlim*scf)*(x>=-axlim*scf)*(y<=axlim*scf)*(y>=-axlim*scf)
                    imp.data[j].calc_phase()
                    phase = npy.where(imp.data[j].phase<=6,imp.data[j].phase-1,imp.data[j].phase)
                    phase = npy.where(phase<2,6,phase)
                    im = plt.scatter(x[select]/scf,y[select]/scf,s=scsize,c=phase[select],ec=None,alpha=1.,cmap=cmapphase,norm=matplotlib.colors.Normalize(vmin=phmin,vmax=phmax,clip=False),rasterized=True)

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
                if l < len(types)-1:
                    plt.gca().set_xticklabels([])
                if m==len(imps)*len(types):
                    if scale=='Mm':
                        plt.xlabel('x (Mm)')
                    elif scale=='km':
                        plt.xlabel('x (km)')
                    elif scale=='earth' or scale=='Earth':
                        plt.xlabel(r'x (R$_\oplus$)')
                    else:
                        plt.xlabel('x')
                plt.xlim(-axlim,axlim)
                plt.ylim(-axlim,axlim)
                plt.minorticks_on()
                if i==0:
                    if type=='density' or type=='rho':
                        if uppercaselab:
                            plt.text(0.05,0.95,chr(64+m),ha='left',va='top',fontsize=9,fontweight='bold',color='w',transform=plt.gca().transAxes)
                        else:
                            plt.text(0.05,0.95,chr(96+m),ha='left',va='top',fontsize=9,fontweight='bold',color='w',transform=plt.gca().transAxes)
                    else:
                        if uppercaselab:
                            plt.text(0.05,0.95,chr(64+m),ha='left',va='top',fontsize=9,fontweight='bold',color='k',transform=plt.gca().transAxes)
                        else:
                            plt.text(0.05,0.95,chr(96+m),ha='left',va='top',fontsize=9,fontweight='bold',color='k',transform=plt.gca().transAxes)
                if type=='density' or type=='rho':
                    if ti<1.1:
                        plt.text(0.96,0.96,'{:.2f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color='w',transform=plt.gca().transAxes)
                    elif ti>23.8:
                        plt.text(0.96,0.96,'{:.0f} hrs'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color='w',transform=plt.gca().transAxes)
                    else:
                        plt.text(0.96,0.96,'{:.1f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color='w',transform=plt.gca().transAxes)
                else:
                    if ti<1.1:
                        plt.text(0.96,0.96,'{:.2f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color='k',transform=plt.gca().transAxes)
                    elif ti>23.8:
                        plt.text(0.96,0.96,'{:.0f} hrs'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color='k',transform=plt.gca().transAxes)
                    else:
                        plt.text(0.96,0.96,'{:.1f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color='k',transform=plt.gca().transAxes)

                ax = plt.gca()
                if i>0:
                    colpos = ax.get_position()
                    #print(colpos.x0,colpos.width)
                    gap = leftcol.x1-colpos.x0
                    ax.set_position([colpos.x0+gap,colpos.y0,colpos.width,colpos.height])
                    colpos = ax.get_position()
                #else:
                #    pos = ax.get_position()
                leftcol = ax.get_position()
                if m>1:
                    rowpos = ax.get_position()
                    gap = toprow.y0-rowpos.y1
                    if k==1:
                        gap -= 0.04/max(1,len(imps)-1.1)#0.05
                    ax.set_position([rowpos.x0,rowpos.y0+gap,rowpos.width,rowpos.height])
                #else:
                #    plt.subplots_adjust(wspace=0)

                if m==1 and i==0:
                    firstplot = plt.gca()
                
                if type!='materials' and type!='mat':
                    if i == len(seq)-1 and (m==1 or m==len(imps)*len(types)):
                        pbox = ax.get_position()
                        xw = 0.33
                        if m==1:
                            cbar_ax = fig.add_axes([pbox.x1-xw, pbox.y1+0.26/max(1,2*len(imps)-1.9)*xw, xw, 0.032/max(1,2*len(imps)-1.8)])#0.27
                            #cbar_ax = fig.add_axes([pbox.x1-xw, pbox.y1+0.32*xw, xw, 0.015])
                        elif m==len(imps)*len(types):
                            pbox = firstplot.get_position()
                            #cbar_ax = fig.add_axes([pbox.x1-xw, pbox.y0-0.6/max(1,1.5*len(imps)-0.7)*xw, xw, 0.032/max(1,2*len(imps)-1.8)])#0.47
                            cbar_ax = fig.add_axes([pbox.x0, pbox.y1+0.26/max(1,2*len(imps)-1.9)*xw, xw, 0.032/max(1,2*len(imps)-1.8)])#0.47
                        cbar = fig.colorbar(im,cax=cbar_ax,orientation='horizontal')
                        ##plt.minorticks_on()
                        if type=='density' or type=='rho':
                            cbar_ax.xaxis.set_label_text(r'Density (g$\,$cm$^{-3}$)')
                        if type=='pressure' or type=='P':
                            cbar_ax.xaxis.set_label_text(r'$P$ (GPa)')
                        elif type in ['entropy','ent','S']:
                            cbar_ax.xaxis.set_label_text(r'$S$ (kJ$\,$K$^{-1}\,$kg$^{-1}$)')
                        elif type=='phase':
                            ##cbar = fig.colorbar(im1, cax=cax, ticks = np.arange(13)/12, orientation='vertical')
                            cbar.ax.set_xticklabels(['','s','s+l','l','l+v','v','scf'])  #  colorbar ['s','s+l','l','l+v']
                            #cbar.ax.set_xticklabels(['s','s+l','l','l+v','v','scf'])  #  colorbar ['s','s+l','l','l+v']
                            cbar_ax.xaxis.set_label_text(r'Phase')
                        
                        cbar_ax.xaxis.set_label_position('top')
            k=0
            m+=1
            l+=1
            cbar_ax = None
            toprow = ax.get_position()
        k=1

    #fig.set_figwidth(10)
    print(colpos.x0,colpos.y0,colpos.x1,colpos.y1)
    plt.show()
    return fig
