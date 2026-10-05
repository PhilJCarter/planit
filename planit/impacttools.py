"""
   planit impact class and analysis tools
"""

from .globaldefs import *
from .snaptools import Snapshot

import numpy as npy
import scipy
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cmasher

# for movie:
from IPython import display
import matplotlib.animation


class ImpSnapshot(Snapshot):
    pass


class Impact:
    """
    Gadget/Swift impact class
    
    A wrapper for holding multiple snapshots
    
    load()
    plotseq()
    """
    def __init__(self):
        #self.targ = None
        #self.proj = None
        #self.v = None
        #self.b = None
        self.nsnaps = 0
        self.snap = self.data = None
        
    def load(self,loc,thermo=False,inter=1,compress=True,code='swift',ndigits=4,prefix='snapshot',sep='_',prefix2=None,flist=None):
        Nf2 = Nf = 0
        increment1 = increment2 = 1
        flist2 = []
        loc = str(loc)
        if loc[-1] != '/':
            loc = loc + '/'
        files2 = npy.empty(0)
        if flist:
            files = files1 = flist
            self.nsnaps = len(flist)
        else:
            if code=='swift':
                flist = sorted(glob.glob(loc+prefix+sep+'[0-9]*.hdf5'))
                if prefix2:
                    flist2 = sorted(glob.glob(loc+prefix2+sep+'[0-9]*.hdf5'))
            else:
                flist = sorted(glob.glob(loc+prefix+sep+'[0-9]*'))
                if prefix2:
                    flist2 = sorted(glob.glob(loc+prefix2+sep+'[0-9]*'))
            Nf1 = [(flist[x].split('/')[-1]).split(sep)[1].split('.')[0] for x in range(len(flist))]
            if prefix2:
                Nf2 = [(flist2[x].split('/')[-1]).split(sep)[1].split('.')[0] for x in range(len(flist2))]
            if len(Nf1)>0:
                Nf1 = sorted(npy.array(Nf1).astype(int))
                increment1 = int(Nf1[1] - Nf1[0])
                Nf = int(Nf1[-1])
            else:
                Nf = 0
            if prefix2 and len(Nf2)>0:
                Nf2 = sorted(npy.array(Nf2).astype(int))
                increment2 = int(Nf2[1]-Nf2[0])
                Nf2 = int(Nf2[-1])
            else:
                Nf2 = 0
            print('Loading', int(Nf/increment1 + Nf2/increment2), 'files...')
            self.nsnaps = len(flist)+len(flist2)
            if self.nsnaps>2:
                if Nf2>0:
                    files2 = npy.arange(0,Nf2+increment2,inter*increment2)
                files1 = npy.arange(0,Nf+increment1,inter*increment1)
                files = npy.append(files1,files2)
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
                if i<len(files1):
                    self.data[i].load(loc+prefix+sep+'{:>0{width}d}.hdf5'.format(int(files[i]),width=ndigits),headonly=honly,thermo=thermo,compress=compress)
                else:
                    self.data[i].load(loc+prefix2+sep+'{:>0{width}d}.hdf5'.format(int(files[i]),width=ndigits),headonly=honly,thermo=thermo,compress=compress)
            else:
                if i<len(files1):
                    self.data[i].load(loc+prefix+sep+'{:>0{width}d}'.format(int(files[i]),width=ndigits),headonly=honly,thermo=thermo,compress=compress)
                else:
                    self.data[i].load(loc+prefix2+sep+'{:>0{width}d}'.format(int(files[i]),width=ndigits),headonly=honly,thermo=thermo,compress=compress)

        self.snap = self.data


    def plotseq(self, n=4, type='materials', seq=None, times=None, scale='Mm', potmin=False, tcut = 3600., zoom=1., focus='potmin'):
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
        Ng = 801j
        Ngz = 21j

        zi=npy.linspace(zmin,zmax,int(Ngz.imag))
        X,Y,Z = npy.mgrid[-axlim:axlim:(Ng),-axlim:axlim:(Ng),zmin:zmax:(Ngz)]

        if potmin:          # for backwards compatability
            focus='potmin'

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

            if type in ['materials','mat','mats','material']:
                im = plot_snapshot_scatter(self.data[j],cmap=cmap,plotQ=-(self.data[j].id/PROJ_ID_OFFSET).astype(int),ptype='mat',axlim=axlim,scf=scf,scale=scale,zcut=zcut,tcut=tcut,focus=focus)
            elif type=='density' or type=='rho':
                im = plot_snapshot_fluid(self.data[j],X,Y,Z,zi,ax=plt.gca(),cmap=cmap,ptype=type,axlim=axlim,scf=scf,scale=scale,zcut=zcut,tcut=tcut,vcut=0.5*self.data[0].vel.min(),focus=focus)
            elif type in ['entropy','ent','S']:        
                im = plot_snapshot_fluid(self.data[j],X,Y,Z,zi,plotQ=self.data[j].S/1e7,ax=plt.gca(),cmap=cmap,ptype=type,axlim=axlim,scf=scf,scale=scale,zcut=zcut,tcut=tcut,vcut=0.5*self.data[0].vel.min(),focus=focus)        
            elif type in ['temperature','T','temp']:        
                im = plot_snapshot_fluid(self.data[j],X,Y,Z,zi,plotQ=self.data[j].T,ax=plt.gca(),cmap=cmap,ptype=type,axlim=axlim,scf=scf,scale=scale,zcut=zcut,tcut=tcut,vcut=0.5*self.data[0].vel.min(),focus=focus)        
            elif type in ['pressure','P']:        
                im = plot_snapshot_fluid(self.data[j],X,Y,Z,zi,plotQ=self.data[j].P/1e9,ax=plt.gca(),cmap=cmap,ptype=type,axlim=axlim,scf=scf,scale=scale,zcut=zcut,tcut=tcut,vcut=0.5*self.data[0].vel.min(),focus=focus)        
            elif type=='phase':
                im = plot_snapshot_scatter(self.data[j],cmap=cmapphase,plotQ=self.data[j].phase,ptype='phase',axlim=axlim,scf=scf,scale=scale,zcut=zcut,tcut=tcut,focus=focus)

            if i>0:
                plt.gca().set_yticklabels([])
                plt.gca().set_ylabel('')
            
            if i == len(seq)-1:
                tlab = plt.gca().texts[0]
                tlab.set_text(tlab.get_text()+' hrs')
   
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
                    elif type in ['pressure','P']:
                        cbar_ax.xaxis.set_label_text(r'$P$ (GPa)')
                    elif type in ['temperature','T','temp']:
                        #locs = ticker.LogLocator(base=10.0, subs=[1.0, 5.0])
                        #cbar.ax.xaxis.set_major_locator(locs)
                        locs = [100,300,1000,3000,10000,30000]
                        cbar.set_ticks(locs)
                        cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:g}'))
                        #cbar.ax.xaxis.set_major_formatter(ticker.LogFormatter(base=10.0, labelOnlyBase=False))#10, labelOnlyBase=False)
                        cbar_ax.xaxis.set_label_text(r'$T$ (K)')
                    elif type=='phase':
                        cbar.set_ticks([3.,4.,5.,6.,7.,8.])
                        #cbar.ax.set_xticklabels(['','s','s+l','l','l+v','v','scf'])  #  colorbar ['s','s+l','l','l+v']
                        cbar.ax.set_xticklabels(['s','s+l','l','l+v','v','scf'])  #  colorbar ['s','s+l','l','l+v']
                        cbar_ax.xaxis.set_label_text(r'Phase')
                    cbar_ax.xaxis.set_label_position('top')
        plt.subplots_adjust(wspace=0, hspace=0)
        #plt.show(block=False)
        return fig


    def attrmov(self, n=40, ptype='materials', seq=None, times=None, scale='Mm', focus='potmin', fps=12, zoom=1., cmap = None, dpi=400, bitrate=5000):
        if not (seq or times): # change to allow numpy array
            if self.nsnaps>n:
                seq = npy.logspace(0,npy.log10(len(self.data)-1),n).astype(int)
            else:
                seq = npy.arange(self.nsnaps-1)
        elif seq:
            n = len(seq)
        
        if ptype not in ['mat','materials','mats','material','density','rho','S','ent','entropy','phase','pressure','P']:
            raise ValueError('Unknown plot type:',ptype)
        
        if scale=='Mm':
            scf = 1e8
        elif scale=='km':
            scf = 1e5
        elif scale=='earth' or scale=='Earth':
            scf = 6.371e8
        else:
            scf = scale
        axlim = 1.5*int(npy.ceil((self.data[0].x.max()-self.data[0].x.min())/scf/8.))/zoom #4*
        zcut = axlim/200.*scf
    
        tcut = 3600.   #time cut for pre-contact treatment
        
        # number of cells for grid
        Ng = 801j
        Ngz = 21j
        #Ng=1281j
        #Ngz=101j
    
        # region to grid/plot
        zmin=-axlim/10.
        zmax=axlim/10.
        
        zlim = zmax/5.*scf
    
        zi=npy.linspace(zmin,zmax,int(Ngz.imag))
        X,Y,Z = npy.mgrid[-axlim:axlim:(Ng),-axlim:axlim:(Ng),zmin:zmax:(Ngz)]
        
        if not cmap:
            if ptype in ['mat', 'materials','mats','material']:
                cmap = plt.get_cmap('cmr.bubblegum')
            elif ptype == ['rho','density']:
                cmap = plt.get_cmap('cmr.eclipse')
            elif ptype == ['S','entropy','ent']:
                cmap = plt.get_cmap('magma')
            elif ptype in ['phase',]:
                cmap = plt.get_cmap('plasma', 6)
            else:
                cmap = plt.get_cmap('plasma') #.copy()
        if ptype in ['phase',]:
            cmap = matplotlib.colors.ListedColormap(cmap.colors[[0,1,2,3,4,5],:])
        if ptype in ['rho', 'density']:        
            cmap.set_under('k')
        else:
            cmap.set_under('w')
                
        j = 0
        def attrmovfunc(j):
            plt.clf()
            plt.minorticks_on()    
    
            if ptype in ['mat','materials','mats','material']:
                #plt.scatter(x[modz<zcut]/scf,y[modz<zcut]/scf,s=0.1,c=-(self.data[j].id/BODYOFF).astype(int)[modz<zcut],vmin=-(self.data[0].id/BODYOFF).astype(int).max(),vmax=0,cmap=cmap)
                #plt.scatter(x[modz<zcut]/scf,y[modz<zcut]/scf,s=0.1,c=(self.data[j].vx)[modz<zcut],alpha=1.)
                im = plot_snapshot_scatter(self.data[j],cmap=cmap,plotQ=-(self.data[j].id/PROJ_ID_OFFSET).astype(int),ax=plt.gca(),ptype=ptype,axlim=axlim,scf=scf,scale=scale,zcut=zcut,tcut=tcut,focus=focus)
            elif ptype in ['phase',] :
                #self.data[j].calc_phase()
                #phase = npy.where(self.data[j].phase<=6,self.data[j].phase-1,self.data[j].phase)
                #phase = npy.where(phase<2,6,phase)
                #im = plt.scatter(x[modz<zcut]/scf,y[modz<zcut]/scf,s=0.1,c=phase[modz<zcut],cmap=cmap,norm=matplotlib.colors.Normalize(vmin=phmin,vmax=phmax,clip=False))
                im = plot_snapshot_scatter(self.data[j],cmap=cmap,plotQ=self.data[j].phase,ax=plt.gca(),ptype=ptype,axlim=axlim,scf=scf,scale=scale,zcut=zcut,tcut=tcut,focus=focus)
            elif ptype in ['rho','density']:        
                im = plot_snapshot_fluid(self.data[j],X,Y,Z,zi,plotQ=self.data[j].rho,ax=plt.gca(),cmap=cmap,ptype=ptype,axlim=axlim,scf=scf,scale=scale,zcut=zcut,tcut=tcut,vcut=0.5*self.data[0].vel.min(),focus=focus)            
            elif ptype in ['ent','S','entropy']:        
                # if self.data[j].header.time<tcut: #100 #500
#                     vcut = 0.5*self.data[0].vel.min() #-5.e5 #8
#                     vit = scipy.interpolate.griddata((x[(self.data[j].vx>vcut)*(modz<zlim)]/scf,y[(self.data[j].vx>vcut)*(modz<zlim)]/scf,z[(self.data[j].vx>vcut)*(modz<zlim)]/scf),self.data[j].S[(self.data[j].vx>vcut)*(modz<zlim)]/1e7,(X,Y,Z),method='linear',fill_value=1.e-18)
#                     rhoit = scipy.interpolate.griddata((x[(self.data[j].vx>vcut)*(modz<zlim)]/scf,y[(self.data[j].vx>vcut)*(modz<zlim)]/scf,z[(self.data[j].vx>vcut)*(modz<zlim)]/scf),self.data[j].rho[(self.data[j].vx>vcut)*(modz<zlim)],(X,Y,Z),method='linear',fill_value=1.e-18)
#                 vi = scipy.interpolate.griddata((x[(self.data[j].vx<vcut)*(modz<zlim)]/scf,y[(self.data[j].vx<vcut)*(modz<zlim)]/scf,z[(self.data[j].vx<vcut)*(modz<zlim)]/scf),self.data[j].S[(self.data[j].vx<vcut)*(modz<zlim)]/1e7,(X,Y,Z),method='linear',fill_value=1.e-18)
#                 rhoi = scipy.interpolate.griddata((x[(self.data[j].vx<vcut)*(modz<zlim)]/scf,y[(self.data[j].vx<vcut)*(modz<zlim)]/scf,z[(self.data[j].vx<vcut)*(modz<zlim)]/scf),self.data[j].rho[(self.data[j].vx<vcut)*(modz<zlim)],(X,Y,Z),method='linear',fill_value=1.e-18)
#                 if self.data[j].header.time<tcut: #100 #500
#                     vi = npy.where(vi>vit,vi,vit)
#                     rhoi = npy.where(rhoi>rhoit,rhoi,rhoit)
#                 if self.data[j].header.time > tcut and focus=='potmin':
#                     coz = (z[self.data[j].pot==self.data[j].pot.min()])[0]/scf #s.z[modz<zcut]
#                 elif focus=='targcore':
#                     coz = npy.median(z[self.data[j].id<planit.BODYOFF])/scf #s.z[modz<zcut]
#                 else:
#                     coz = 0 #(z[self.data[j].pot==self.data[j].pot.min()])[0] #s.z[modz<zcut]
#                 #print(j,npy.median(self.data[j].z[self.data[j].id<planit.BODYOFF]),coz)
#                 nn=(npy.nonzero(zi==(zi[zi<=coz])[-1])[0])[0]
#                 alphas=matplotlib.colors.LogNorm(vmin=0.05*rhomin,vmax=rhomax,clip=True)(rhoi[:,:,nn].T)
#                 cols = matplotlib.colors.Normalize(vmin=cmin,vmax=cmax,clip=True)(vi[:,:,nn].T)
#                 cols = cmap(cols)
#                 cols[..., -1] = alphas    
#                 ax = plt.gca()
#                 im = ax.imshow(cols,origin='lower',extent=[-axlim,axlim,-axlim,axlim],vmin=cmin,vmax=cmax,cmap=cmap)
                im = plot_snapshot_fluid(self.data[j],X,Y,Z,zi,plotQ=self.data[j].S/1e7,ax=plt.gca(),cmap=cmap,ptype=ptype,axlim=axlim,scf=scf,scale=scale,zcut=zcut,tcut=tcut,vcut=0.5*self.data[0].vel.min(),focus=focus)            
            elif ptype in ['pressure','P']:        
                im = plot_snapshot_fluid(self.data[j],X,Y,Z,zi,plotQ=self.data[j].P/1e9,ax=plt.gca(),cmap=cmap,ptype=ptype,axlim=axlim,scf=scf,scale=scale,zcut=zcut,tcut=tcut,vcut=0.5*self.data[0].vel.min(),focus=focus)            
        
            #ax=plt.gca()
            #ax.xaxis.set_major_locator(ticker.MultipleLocator(10.00))
            #ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.00))
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(10.00))
            #ax.yaxis.set_minor_locator(ticker.MultipleLocator(2.00))
    
            if ptype not in ['mat', 'materials','mats','material']:
                pbox=plt.gca().get_position()
                cbar_ax = fig.add_axes([pbox.x1*1.02, pbox.y0, 0.035, 1.0*(pbox.y1-pbox.y0)])
                cbar = fig.colorbar(im,cax=cbar_ax,orientation='vertical')
                if ptype in ['rho','density']:
                    cbar_ax.yaxis.set_label_text(r'$\rho$ (g$\,$cm$^{-3}$)')
                elif ptype in ['ent', 'entropy', 'S']:
                    cbar_ax.yaxis.set_label_text(r'$S$ (kJ$\,$K$^{-1}\,$kg$^{-1}$)')
                elif ptype in ['P','pressure']:
                    cbar_ax.yaxis.set_label_text(r'$P$ (GPa)')
                elif ptype in ['phase',]:
                    ##cbar = fig.colorbar(im1, cax=cax, ticks = np.arange(13)/12, orientation='vertical')
                    #cbar.ax.set_xticklabels(['','s','s+l','l','l+v'])  #  colorbar ['s','s+l','l','l+v']
                    cbar.set_ticks([3.,4.,5.,6.,7.,8.])
                    cbar.ax.set_yticklabels(['s',' s+l',' l',' l+v',' v',' scf'])  #  colorbar ['s','s+l','l','l+v']
                    cbar_ax.yaxis.set_label_text(r'Phase')
                cbar_ax.yaxis.set_label_position('right')
                
    
            if n>50 and j!=0 and j!=len(self.data)-1:
                self.data[j].freedata()
    
    
        fig=plt.figure(figsize=(3.5,2.8))
    
        if len(seq)==1:
            attrmovfunc(seq[0])
            plt.subplots_adjust(top=0.98,bottom=0.11,left=0.165,right=0.796)
            plt.show(block=False)
            return fig
        else:
            if ptype in ['mat', 'materials']:
                movfile = self.data[0].file.strip(self.data[0].file.split('/')[-1])+'materials.mp4'
            elif ptype in ['rho','density']:
                movfile = self.data[0].file.strip(self.data[0].file.split('/')[-1])+'density.mp4'
            elif ptype in ['ent', 'entropy', 'S']:
                movfile = self.data[0].file.strip(self.data[0].file.split('/')[-1])+'entropy.mp4'
            elif ptype in ['phase',]:
                movfile = self.data[0].file.strip(self.data[0].file.split('/')[-1])+'phase.mp4'
            elif ptype in ['pressure','P']:
                movfile = self.data[0].file.strip(self.data[0].file.split('/')[-1])+'pressure.mp4'
            else:
                movfile = self.data[0].file.strip(self.data[0].file.split('/')[-1])+'attrmov.mp4'
            anim=matplotlib.animation.FuncAnimation(fig,attrmovfunc,seq)
            plt.subplots_adjust(top=0.975,bottom=0.105,left=0.166,right=0.796)
            ##plt.subplots_adjust(top=0.98,bottom=0.1,left=0.135,right=0.795)
            ##anim.save(movfile,dpi=200,writer=matplotlib.animation.PillowWriter(fps=fps,bitrate=1000)) #writer='ffmpeg' # 800 # 9000 fps=fps,bitrate=1000
            anim.save(movfile,dpi=dpi,fps=fps,bitrate=bitrate) #writer='ffmpeg' # 800 # 9000 fps=fps,bitrate=1000
            plt.show()
            plt.close()
            
            html=display.Video(movfile,embed=True,width=500)
            return html
        



def multiplotseq(imps, n=4, types='materials', seqs=None, times=None, scale='Mm', potmin=False, zoom=1.,uppercaselab=False,focus='potmin'):
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
    Ng = 1801j
    Ngz = 15j

    scsize = 1.5

    if potmin:
        focus='potmin'


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
    fig.set_figheight(8./max(n)*(len(imps)*len(types)) + 1.*len(imps)**1.75)#-0.1
    gs = fig.add_gridspec(ncols=max(n), nrows=len(imps)*len(types),wspace=0,hspace=0,right=0.98,top=0.96)
    #print(max(n),len(imps)*len(types))
    
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
                #ti = ( imp.data[j].header.time )/3600.
                #if npy.ndim(ti)>0:
                #    ti=ti[0]
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

                if imp.data[j].header.time < tcut:
                    vcut = 0.5*imp.data[0].vel.min()

                if type in ['materials','mat','mats','material']:
                    select = (modz<zcut)*(x<=axlim*scf)*(x>=-axlim*scf)*(y<=axlim*scf)*(y>=-axlim*scf)
                    plt.scatter(x[select]/scf,y[select]/scf,s=scsize,c=(imp.data[j].id/BODYOFF).astype(int)[select],alpha=1.,cmap=cmap.reversed(),vmax=(imp.data[0].id/BODYOFF).astype(int).max(),vmin=0,ec=None,rasterized=True)
                    #print(npy.unique(-(imp.data[j].id/BODYOFF).astype(int)[modz<zcut]))
                elif type=='density' or type=='rho':        
                    #if imp.data[j].header.time<tcut:
                    #    vcut = 0.5*imp.data[0].vel.min() #-5.e5 #8
                        #rhoit = scipy.interpolate.griddata((x[(imp.data[j].vx>vcut)*(modz<zlim)]/scf,y[(imp.data[j].vx>vcut)*(modz<zlim)]/scf,z[(imp.data[j].vx>vcut)*(modz<zlim)]/scf),imp.data[j].rho[(imp.data[j].vx>vcut)*(modz<zlim)],(X,Y,Z),method='linear',fill_value=1.e-18)
                    #rhoi = scipy.interpolate.griddata((x[(imp.data[j].vx<vcut)*(modz<zlim)]/scf,y[(imp.data[j].vx<vcut)*(modz<zlim)]/scf,z[(imp.data[j].vx<vcut)*(modz<zlim)]/scf),imp.data[j].rho[(imp.data[j].vx<vcut)*(modz<zlim)],(X,Y,Z),method='linear',fill_value=1.e-18)
                    #if imp.data[j].header.time<tcut:
                    #    rhoi = npy.where(rhoi>rhoit,rhoi,rhoit)
                    #if imp.data[j].header.time != 0 and potmin:
                    #    coz = (z[imp.data[j].pot==imp.data[j].pot.min()])[0]/scf #s.z[modz<zcut]
                    #else:
                    #    coz = 0 #(z[imp.data[j].pot==imp.data[j].pot.min()])[0] #s.z[modz<zcut]
                    #nn=(npy.nonzero(zi==(zi[zi<=coz])[-1])[0])[0]
                    #cols = rhoi[:,:,nn].T
                    ##cols=matplotlib.colors.LogNorm(vmin=rhomin,vmax=rhomax,clip=False)(rhoi[:,:,nn].T)
                    ##cols=cmap(cols)
                    ax = plt.gca()
                    #im = ax.imshow(cols,origin='lower',extent=[-axlim,axlim,-axlim,axlim],cmap=cmap,rasterized=True,norm=matplotlib.colors.LogNorm(vmin=rhomin,vmax=rhomax,clip=False))#vmin=rhomin,vmax=rhomax
                    im = plot_snapshot_fluid(imp.data[j],X,Y,Z,zi,ax=ax,cmap=cmap,ptype=type,axlim=axlim,scf=scf,scale=scale,zcut=zcut,tcut=tcut,vcut=vcut,focus=focus)
                    #ax.tick_params(colors='w',which='both',labelcolor='k')
                    #ax.spines['top'].set_color('w')
                    #ax.spines['bottom'].set_color('w')
                    #ax.spines['left'].set_color('w')
                    #ax.spines['right'].set_color('w')

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
                    #ax = plt.gca()
                    im = plot_snapshot_fluid(imp.data[j],X,Y,Z,zi,ax=plt.gca(),cmap=cmap,ptype=type,plotQ=imp.data[j].S/1e7,axlim=axlim,scf=scf,zcut=zcut,focus='potmin',tcut=tcut,vcut=vcut)
                    #im = ax.imshow(cols,origin='lower',extent=[-axlim,axlim,-axlim,axlim],vmin=cmin,vmax=cmax,cmap=cmap)
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
                #plt.xlim(-axlim,axlim)
                #plt.ylim(-axlim,axlim)
                #plt.minorticks_on()
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
                #if type=='density' or type=='rho':
                #    if ti<1.1:
                #        plt.text(0.96,0.96,'{:.2f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color='w',transform=plt.gca().transAxes)
                #    elif ti>23.8:
                #        plt.text(0.96,0.96,'{:.0f} hrs'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color='w',transform=plt.gca().transAxes)
                #    else:
                #        plt.text(0.96,0.96,'{:.1f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color='w',transform=plt.gca().transAxes)
                #else:
                #    if ti<1.1:
                #        plt.text(0.96,0.96,'{:.2f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color='k',transform=plt.gca().transAxes)
                #    elif ti>23.8:
                #        plt.text(0.96,0.96,'{:.0f} hrs'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color='k',transform=plt.gca().transAxes)
                #    else:
                #        plt.text(0.96,0.96,'{:.1f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color='k',transform=plt.gca().transAxes)

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
                
                if type not in ['materials','mat','mats','material']:
                    if i == len(seq)-1 and (m==1 or m==len(imps)*len(types)):
                        pbox = ax.get_position()
                        xw = 0.33
                        if m==1:
                            cbar_ax = fig.add_axes([pbox.x1-xw, pbox.y1+0.26/max(1,2*len(imps)-1.9)*xw, xw, 0.02/max(1,2*len(imps)**0.8-1.8)])#0.27
                            #cbar_ax = fig.add_axes([pbox.x1-xw, pbox.y1+0.32*xw, xw, 0.015])
                        elif m==len(imps)*len(types):
                            pbox = firstplot.get_position()
                            #cbar_ax = fig.add_axes([pbox.x1-xw, pbox.y0-0.6/max(1,1.5*len(imps)-0.7)*xw, xw, 0.032/max(1,2*len(imps)-1.8)])#0.47
                            cbar_ax = fig.add_axes([pbox.x0, pbox.y1+0.26/max(1,2*len(imps)-1.9)*xw, xw, 0.02/max(1,2*len(imps)**0.8-1.8)])#0.47
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
                            cbar.ax.set_xticks([3,4,5,6,7,8])
                            cbar.ax.set_xticklabels(['s','s+l','l','l+v','v','scf'])  #  colorbar ['s','s+l','l','l+v']
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
    #print(colpos.x0,colpos.y0,colpos.x1,colpos.y1)
    #plt.show()
    return fig


def plot_snapshot_scatter(snap,ax=plt.gca(),cmap=plt.get_cmap('plasma'),plotQ=None,ptype='mat',axlim=1.,scf=1.,scale='Earth',zcut=0.,tcut=0.,focus='potmin',rhomin=1e-5):
    #phase flag limits
    phmin=2.5
    phmax=8.5

    ti = (snap.header.time)/3600.
    if npy.ndim(ti)>0:
        ti=ti[0]

    if focus=='potmin' and snap.header.time >= tcut:
        x = snap.x - (snap.x[snap.pot==snap.pot.min()])[0]
        y = snap.y - (snap.y[snap.pot==snap.pot.min()])[0]
        z = snap.z - (snap.z[snap.pot==snap.pot.min()])[0]
    elif focus=='targcore':
        x = snap.x - npy.median(snap.x[snap.id<PROJ_ID_OFFSET])
        y = snap.y - npy.median(snap.y[snap.id<PROJ_ID_OFFSET])
        z = snap.z - npy.median(snap.z[snap.id<PROJ_ID_OFFSET])
    else:
        x = snap.x
        y = snap.y
        z = snap.z
    modz = npy.abs(z)

    # reorder phases for plotting
    if ptype in ['phase',]:
        plotQ = npy.where(plotQ<=6,plotQ-1,plotQ)
        plotQ = npy.where(plotQ<2,6,plotQ)
        cmin = phmin
        cmax = phmax
        norm = matplotlib.colors.Normalize(vmin=cmin,vmax=cmax,clip=False)
    elif ptype in ['mat','materials','mats','material']:
        #cmin = 0
        #cmax = 1
        norm = None
    im = plt.scatter(x[modz<zcut]/scf,y[modz<zcut]/scf,s=0.1,c=plotQ[modz<zcut],alpha=1.,cmap=cmap,norm=norm,rasterized=True) #s=0.8

    labcolor='k'
    if ti<1.1:
        plt.text(0.96,0.96,'{:.2f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color=labcolor,transform=plt.gca().transAxes)
    elif ti>23.8:
        plt.text(0.96,0.96,'{:.0f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color=labcolor,transform=plt.gca().transAxes)
    else:
        plt.text(0.96,0.96,'{:.1f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color=labcolor,transform=plt.gca().transAxes)

    plt.xlim(-axlim,axlim)
    plt.ylim(-axlim,axlim)
    ax.set_aspect('equal')
    plt.minorticks_on()        

    if scale=='Mm':
        plt.ylabel('y (Mm)')
    elif scale=='km':
        plt.ylabel('y (km)')
    elif scale in ['earth','Earth']:
        plt.ylabel(r'y (R$_\oplus$)')
    else:
        plt.ylabel('y')
    if scale=='Mm':
        plt.xlabel('x (Mm)')
    elif scale=='km':
        plt.xlabel('x (km)')
    elif scale in ['earth','Earth']:
        plt.xlabel(r'x (R$_\oplus$)')
    else:
        plt.xlabel('x')

    return im


def plot_snapshot_fluid(snap,X=npy.empty([]),Y=npy.empty([]),Z=npy.empty([]),zi=npy.empty([]),ax=plt.gca(),cmap=plt.get_cmap('plasma'),plotQ=None,ptype='rho',axlim=1.,scf=1.,scale='Earth',zcut=0.,tcut=0.,vcut=None,rhomin=1e-5,focus='potmin'):
    # density limits
    # rhomin=1e-5
    rhomax=10.
    # entropy limits
    Smin=1.5
    Smax=10.
    # temperature limits
    Tmin=100.
    Tmax=30000.
    # pressure limits
    Pmin=1.e-9
    Pmax=1000.
    vi = vit = None

    # number of cells for grid
    Ng = 801j
    Ngz = 21j
    if zi.ndim ==0:
        zi=npy.linspace(zmin,zmax,int(Ngz.imag))
    if X.ndim==0 or Y.ndim==0 or Z.ndim==0:
        X,Y,Z = npy.mgrid[-axlim:axlim:(Ng),-axlim:axlim:(Ng),zmin:zmax:(Ngz)]

    ti = (snap.header.time)/3600.
    if npy.ndim(ti)>0:
        ti=ti[0]

    if focus=='potmin' and snap.header.time >= tcut:
        x = snap.x - (snap.x[snap.pot==snap.pot.min()])[0]
        y = snap.y - (snap.y[snap.pot==snap.pot.min()])[0]
        z = snap.z - (snap.z[snap.pot==snap.pot.min()])[0]
    elif focus=='targcore':
        x = snap.x - npy.median(snap.x[snap.id<PROJ_ID_OFFSET])
        y = snap.y - npy.median(snap.y[snap.id<PROJ_ID_OFFSET])
        z = snap.z - npy.median(snap.z[snap.id<PROJ_ID_OFFSET])
    else:
        x = snap.x
        y = snap.y
        z = snap.z
    modz = npy.abs(z)

    if snap.header.time<tcut and vcut:
        if ptype not in ['rho', 'density']:
            vit = scipy.interpolate.griddata((x[(snap.vx>vcut)*(modz<zcut)]/scf,y[(snap.vx>vcut)*(modz<zcut)]/scf,z[(snap.vx>vcut)*(modz<zcut)]/scf),plotQ[(snap.vx>vcut)*(modz<zcut)],(X,Y,Z),method='linear',fill_value=1.e-18)
        rhoit = scipy.interpolate.griddata((x[(snap.vx>vcut)*(modz<zcut)]/scf,y[(snap.vx>vcut)*(modz<zcut)]/scf,z[(snap.vx>vcut)*(modz<zcut)]/scf),snap.rho[(snap.vx>vcut)*(modz<zcut)],(X,Y,Z),method='linear',fill_value=1.e-18)
    else:
        vcut = 2*snap.vel.max()
    if ptype not in ['rho', 'density']:
        vi = scipy.interpolate.griddata((x[(snap.vx<vcut)*(modz<zcut)]/scf,y[(snap.vx<vcut)*(modz<zcut)]/scf,z[(snap.vx<vcut)*(modz<zcut)]/scf),plotQ[(snap.vx<vcut)*(modz<zcut)],(X,Y,Z),method='linear',fill_value=1.e-18)
    rhoi = scipy.interpolate.griddata((x[(snap.vx<vcut)*(modz<zcut)]/scf,y[(snap.vx<vcut)*(modz<zcut)]/scf,z[(snap.vx<vcut)*(modz<zcut)]/scf),snap.rho[(snap.vx<vcut)*(modz<zcut)],(X,Y,Z),method='linear',fill_value=1.e-18)
    if snap.header.time<tcut:
        if vi is not None:
            vi = npy.where(vi>vit,vi,vit)
        rhoi = npy.where(rhoi>rhoit,rhoi,rhoit)
    if snap.header.time >= tcut and focus=='potmin':
        coz = (z[snap.pot==snap.pot.min()])[0] #s.z[modz<zcut]
    else:
        coz = 0
    nn = (npy.nonzero(zi==(zi[zi<=coz])[-1])[0])[0]
    norm = matplotlib.colors.LogNorm(vmin=rhomin,vmax=rhomax,clip=False)
    if vi is not None:
        if ptype in ['P','pressure']:
            cmin = Pmin
            cmax = Pmax
            norm = matplotlib.colors.LogNorm(vmin=cmin,vmax=cmax,clip=False)
            cols = vi[:,:,nn].T #matplotlib.colors.LogNorm(vmin=cmin,vmax=cmax,clip=True)(vi[:,:,nn].T)
        elif ptype in ['ent','S','entropy']:
            cmin = Smin
            cmax = Smax
            norm = matplotlib.colors.Normalize(vmin=cmin,vmax=cmax,clip=False)
            cols = matplotlib.colors.Normalize(vmin=cmin,vmax=cmax,clip=True)(vi[:,:,nn].T)
            alphas = matplotlib.colors.LogNorm(vmin=0.05*rhomin,vmax=rhomax,clip=True)(rhoi[:,:,nn].T)
            cols=cmap(cols)
            cols[..., -1] = alphas
        elif ptype in ['temperature','T','temp']:
            cmin = Tmin
            cmax = Tmax
            norm = matplotlib.colors.LogNorm(vmin=cmin,vmax=cmax,clip=False)
            cols = vi[:,:,nn].T #matplotlib.colors.LogNorm(vmin=cmin,vmax=cmax,clip=True)(vi[:,:,nn].T)
    else:
        cols = rhoi[:,:,nn].T

    im = ax.imshow(cols,origin='lower',extent=[-axlim,axlim,-axlim,axlim],cmap=cmap,norm=norm)

    if ptype in ['rho', 'density']:
        ax.tick_params(colors='w',which='both',labelcolor='k')
        ax.spines['top'].set_color('w')
        ax.spines['bottom'].set_color('w')
        ax.spines['left'].set_color('w')
        ax.spines['right'].set_color('w')

    if ptype in ['rho', 'density']:
        labcolor = 'w'
    else:
        labcolor='k'
    if ti<1.1:
        plt.text(0.96,0.96,'{:.2f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color=labcolor,transform=plt.gca().transAxes)
    elif ti>23.8:
        plt.text(0.96,0.96,'{:.0f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color=labcolor,transform=plt.gca().transAxes)
    else:
        plt.text(0.96,0.96,'{:.1f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color=labcolor,transform=plt.gca().transAxes)

    plt.xlim(-axlim,axlim)
    plt.ylim(-axlim,axlim)
    ax.set_aspect('equal')
    plt.minorticks_on()
      
    if scale=='Mm':
        plt.ylabel('y (Mm)')
    elif scale=='km':
        plt.ylabel('y (km)')
    elif scale in ['earth','Earth']:
        plt.ylabel(r'y (R$_\oplus$)')
    else:
        plt.ylabel('y')
    if scale=='Mm':
        plt.xlabel('x (Mm)')
    elif scale=='km':
        plt.xlabel('x (km)')
    elif scale in ['earth','Earth']:
        plt.xlabel(r'x (R$_\oplus$)')
    else:
        plt.xlabel('x')

    return im
           