"""
   planit impact class and analysis tools
"""

from .globaldefs import *
from .snaptools import Snapshot
from . import plotting

import numpy as npy
import scipy
import glob
import os
import matplotlib
import matplotlib.pyplot as plt
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
    attrmov()
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


    def plotseq(self, n=4, ptype='materials', type=None, seq=None, times=None, scale='Mm', potmin=False, tcut = 3600., zoom=1., focus='potmin', cmap=None):
        """
        tcut -- time cut for pre-contact treatment
        """
        if seq is None: # and times is None:
            if self.nsnaps>n:
                seq = npy.logspace(0,npy.log10(len(self.data)-1),n).astype(int)
            else:
                seq = npy.arange(self.nsnaps)
        elif seq is not None:
            n = len(seq)
        
        if type:          # for backwards compatibility
            ptype = type
        
        scf = plotting.get_scale_fac(scale)
        axlim = 4*int(npy.ceil((self.data[0].x.max()-self.data[0].x.min())/scf/8.)) 
        zcut = axlim/200.*scf

        # region to grid/plot
        #zmin=-axlim/10.
        #zmax=axlim/10.

        axlim /= zoom

        # number of cells for grid
        Ng = 1601j
        #Ngz = 21j

        #zi=npy.linspace(zmin,zmax,int(Ngz.imag))
        #X,Y,Z = npy.mgrid[-axlim:axlim:(Ng),-axlim:axlim:(Ng),zmin:zmax:(Ngz)]
        X,Y = npy.mgrid[-axlim:axlim:(Ng),-axlim:axlim:(Ng)]
        Z = npy.full_like(X, 0)

        if potmin:          # for backwards compatibility
            focus='potmin'

        if not cmap:
            cmap = plotting.get_default_colormap(ptype)
        if ptype in plotting.phase_type:
            cmap = matplotlib.colors.ListedColormap(cmap.colors[[0,1,2,3,4,5],:])
            #cmap = matplotlib.colors.ListedColormap(cmap.colors[0,1,2,3,4,5])
    
        fig = plt.figure(figsize=(10,6))
    
        for i in range(len(seq)):
            j = seq[i]
            plt.subplot(1,n,i+1,aspect='equal')

            if ptype in plotting.materials_type:
                im = plotting.plot_snapshot_scatter(self.data[j],cmap=cmap,plotQ=-(self.data[j].id/PROJ_ID_OFFSET).astype(int),ptype=ptype,axlim=axlim,scale=scale,zcut=zcut,tcut=tcut,focus=focus)
            elif ptype in plotting.density_type:
                im = plotting.plot_snapshot_fluid(self.data[j],X,Y,Z,ax=plt.gca(),cmap=cmap,ptype=ptype,axlim=axlim,scale=scale,zcut=zcut,tcut=tcut,vcut=0.5*self.data[0].vel.min(),focus=focus)
            elif ptype in plotting.entropy_type:        
                im = plotting.plot_snapshot_fluid(self.data[j],X,Y,Z,plotQ=self.data[j].S/1e7,ax=plt.gca(),cmap=cmap,ptype=ptype,axlim=axlim,scale=scale,zcut=zcut,tcut=tcut,vcut=0.5*self.data[0].vel.min(),focus=focus)        
            elif ptype in plotting.temperature_type:        
                im = plotting.plot_snapshot_fluid(self.data[j],X,Y,Z,plotQ=self.data[j].T,ax=plt.gca(),cmap=cmap,ptype=ptype,axlim=axlim,scale=scale,zcut=zcut,tcut=tcut,vcut=0.5*self.data[0].vel.min(),focus=focus)        
            elif ptype in plotting.pressure_type:        
                im = plotting.plot_snapshot_fluid(self.data[j],X,Y,Z,plotQ=self.data[j].P/1e9,ax=plt.gca(),cmap=cmap,ptype=ptype,axlim=axlim,scale=scale,zcut=zcut,tcut=tcut,vcut=0.5*self.data[0].vel.min(),focus=focus)        
            elif ptype in plotting.phase_type:
                im = plotting.plot_snapshot_scatter(self.data[j],cmap=cmap,plotQ=self.data[j].phase,ptype=ptype,axlim=axlim,scale=scale,zcut=zcut,tcut=tcut,focus=focus)

            if i>0:
                plt.gca().set_yticklabels([])
                plt.gca().set_ylabel('')
            
            if i == len(seq)-1:
                tlab = plt.gca().texts[0]
                tlab.set_text(tlab.get_text()+' hrs')
   
            if ptype not in plotting.materials_type:
                if i == len(seq)-1:
                    pbox=plt.gca().get_position()
                    xw = 0.25
                    cbar_ax = fig.add_axes([pbox.x1-xw, pbox.y1*1.1, xw, 0.015])
                    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
                    plotting.format_colorbar_axis(cbar, ptype)
        plt.subplots_adjust(wspace=0, hspace=0)
        #plt.show(block=False)
        return fig


    def attrmov(self, n=40, ptype='materials', seq=None, times=None, scale='Mm', focus='potmin', fps=12, zoom=1., cmap = None, dpi=400, bitrate=5000, tcut = 3600.):
        if seq is None: #and times is None:
            if self.nsnaps>n:
                seq = npy.logspace(0,npy.log10(len(self.data)-1),n).astype(int)
            else:
                seq = npy.arange(self.nsnaps-1)
        elif seq is not None:
            n = len(seq)
        
        if ptype not in [*plotting.materials_type,*plotting.density_type,*plotting.entropy_type,*plotting.phase_type,*plotting.pressure_type,*plotting.temperature_type]:
            raise ValueError('Unknown plot type:',ptype)
        
        scf = plotting.get_scale_fac(scale)
        axlim = 1.5*int(npy.ceil((self.data[0].x.max()-self.data[0].x.min())/scf/8.))/zoom #4*
        zcut = axlim/200.*scf

        # number of cells for grid
        Ng = 1201j
        #Ngz = 21j

        # region to grid/plot
        #zmin=-axlim/10.
        #zmax=axlim/10.
        #zlim = zmax/5.*scf

        #zi=npy.linspace(zmin,zmax,int(Ngz.imag))
        #X,Y,Z = npy.mgrid[-axlim:axlim:(Ng),-axlim:axlim:(Ng),zmin:zmax:(Ngz)]
        X,Y = npy.mgrid[-axlim:axlim:(Ng),-axlim:axlim:(Ng)]
        Z = npy.full_like(X, 0)

        
        if not cmap:
            cmap = plotting.get_default_colormap(ptype)
        if ptype in plotting.phase_type:
            cmap = matplotlib.colors.ListedColormap(cmap.colors[[0,1,2,3,4,5],:])
                
        j = 0
        def attrmovfunc(j):
            plt.clf()
            plt.minorticks_on()    
    
            if ptype in plotting.materials_type:
                im = plotting.plot_snapshot_scatter(self.data[j],cmap=cmap,plotQ=-(self.data[j].id/PROJ_ID_OFFSET).astype(int),ax=plt.gca(),ptype=ptype,axlim=axlim,scale=scale,zcut=zcut,tcut=tcut,focus=focus)
            elif ptype in plotting.phase_type:
                im = plotting.plot_snapshot_scatter(self.data[j],cmap=cmap,plotQ=self.data[j].phase,ax=plt.gca(),ptype=ptype,axlim=axlim,scale=scale,zcut=zcut,tcut=tcut,focus=focus)
            elif ptype in plotting.density_type:        
                im = plotting.plot_snapshot_fluid(self.data[j],X,Y,Z,plotQ=self.data[j].rho,ax=plt.gca(),cmap=cmap,ptype=ptype,axlim=axlim,scale=scale,zcut=zcut,tcut=tcut,vcut=0.5*self.data[0].vel.min(),focus=focus)            
            elif ptype in plotting.entropy_type:        
                im = plotting.plot_snapshot_fluid(self.data[j],X,Y,Z,plotQ=self.data[j].S/1e7,ax=plt.gca(),cmap=cmap,ptype=ptype,axlim=axlim,scale=scale,zcut=zcut,tcut=tcut,vcut=0.5*self.data[0].vel.min(),focus=focus)            
            elif ptype in plotting.temperature_type:        
                im = plotting.plot_snapshot_fluid(self.data[j],X,Y,Z,plotQ=self.data[j].T,ax=plt.gca(),cmap=cmap,ptype=ptype,axlim=axlim,scale=scale,zcut=zcut,tcut=tcut,vcut=0.5*self.data[0].vel.min(),focus=focus)        
            elif ptype in plotting.pressure_type:        
                im = plotting.plot_snapshot_fluid(self.data[j],X,Y,Z,plotQ=self.data[j].P/1e9,ax=plt.gca(),cmap=cmap,ptype=ptype,axlim=axlim,scale=scale,zcut=zcut,tcut=tcut,vcut=0.5*self.data[0].vel.min(),focus=focus)            
    
            if ptype not in plotting.materials_type:
                pbox = plt.gca().get_position()
                cbar_ax = fig.add_axes([pbox.x1*1.02, pbox.y0, 0.035, 1.0*(pbox.y1-pbox.y0)])
                cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
                plotting.format_colorbar_axis(cbar, ptype, y=True)

            if n>30 and j!=0 and j!=len(self.data)-1:
                self.data[j].freedata()

        fig = plt.figure(figsize=(3.5,2.8))
    
        if len(seq)==1:
            attrmovfunc(seq[0])
            plt.subplots_adjust(top=0.98,bottom=0.11,left=0.17,right=0.8)
            plt.show(block=False)
            return fig
        else:
            if matplotlib.animation.FFMpegWriter.isAvailable():
                filetype = '.mp4'
            else:
                filetype = '.gif'
            if ptype in plotting.materials_type:
                movfile = os.path.dirname(self.data[0].file)+'/materials'+filetype
            elif ptype in plotting.density_type:
                movfile = os.path.dirname(self.data[0].file)+'/density'+filetype
            elif ptype in plotting.entropy_type:
                movfile = os.path.dirname(self.data[0].file)+'/entropy'+filetype
            elif ptype in plotting.phase_type:
                movfile = os.path.dirname(self.data[0].file)+'/phase'+filetype
            elif ptype in plotting.pressure_type:
                movfile = os.path.dirname(self.data[0].file)+'/pressure'+filetype
            elif ptype in plotting.temperature_type:
                movfile = os.path.dirname(self.data[0].file)+'/temperature'+filetype
            else:
                movfile = os.path.dirname(self.data[0].file)+'/attrmov'+filetype
            anim=matplotlib.animation.FuncAnimation(fig,attrmovfunc,seq)
            plt.subplots_adjust(top=0.975,bottom=0.105,left=0.17,right=0.8)
            ##anim.save(movfile,dpi=200,writer=matplotlib.animation.PillowWriter(fps=fps,bitrate=1000)) #writer='ffmpeg' # 800 # 9000 fps=fps,bitrate=1000
            anim.save(movfile,dpi=dpi,fps=fps,bitrate=bitrate) #writer='ffmpeg' # 800 # 9000 fps=fps,bitrate=1000
            plt.show()
            plt.close()
            
            html=display.Video(movfile,embed=True,width=500)
            return html



def multiplotseq(imps, n=4, types='materials', seqs=None, times=None, scale='Mm', potmin=False, zoom=1., uppercaselab=False, focus='potmin', tcut=3600.):
    if (not isinstance(imps,list)) and (not isinstance(types,list)):
        return imps.plotseq(n=n,ptype=types,seq=seqs,times=times,scale=scale,focus=focus,zoom=zoom)
    elif not isinstance(imps,list):
        imps=[imps,]
    elif not isinstance(types,list):
        types=[types,]
    if seqs is None: # and times is None:
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
 
    scf = plotting.get_scale_fac(scale)       

    # number of cells for grid
    Ng = 1201j
    #Ngz = 15j

    if potmin:
        focus='potmin'
    
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
        zcut = axlim/200.*scf #200
        # region to grid/plot
        #zmin = -axlim/10.
        #zmax = axlim/10.
        axlim /= zoomfac

        #zlim = zmax/5.*scf
        #zi = npy.linspace(zmin,zmax,int(Ngz.imag))
        #X,Y,Z = npy.mgrid[-axlim:axlim:(Ng),-axlim:axlim:(Ng),zmin:zmax:(Ngz)]
        X,Y = npy.mgrid[-axlim:axlim:(Ng),-axlim:axlim:(Ng)]
        Z = npy.full_like(X, 0)
        
        l = 0

        for ptype in types:
            n = len(seq)
#             if l==0:
#                 cmap = plt.get_cmap('cmr.bubblegum').copy() #'plasma'
#                 cmapphase = plt.get_cmap('plasma', 6).copy()
#             elif l==1 and len(types)>2:
#                 cmap = plt.get_cmap('cmr.bubblegum').copy()#cmocean.cm.haline #plt.get_cmap('magma')#.reversed()
#                 cmapphase = plt.get_cmap('cmr.bubblegum', 6).copy()
#             elif l==2 or l==1:
#                 cmap = plt.get_cmap('cmr.eclipse').copy()
#                 cmapphase = plt.get_cmap('cmr.eclipse', 6).copy()
#             else:
#                 cmap=plt.get_cmap('plasma').copy()
#                 cmapphase = plt.get_cmap('plasma', 6).copy()
            cmap = plotting.get_default_colormap(ptype)
            if ptype in plotting.phase_type:
                cmap = matplotlib.colors.ListedColormap(cmap.colors[[0,1,2,3,4,5],:])
#             if ptype in plotting.density_type:        
#                 cmap.set_under('k')
#             else:
#                 cmap.set_under('w')

            for i in range(len(seq)):
                j = seq[i]
                #plt.subplot(len(imps),n,(m-1)*n + i+1,aspect='equal')
                fig.add_subplot(gs[m-1,i],aspect='equal')

                if imp.data[j].header.time < tcut:
                    vcut = 0.5*imp.data[0].vel.min()

                if ptype in plotting.materials_type:
                    im = plotting.plot_snapshot_scatter(imp.data[j],cmap=cmap,plotQ=-(imp.data[j].id/PROJ_ID_OFFSET).astype(int),ax=plt.gca(),ptype=ptype,axlim=axlim,scale=scale,zcut=zcut,tcut=tcut,focus=focus)
                elif ptype in plotting.density_type:        
                    im = plotting.plot_snapshot_fluid(imp.data[j],X,Y,Z,ax=plt.gca(),cmap=cmap,ptype=ptype,plotQ=imp.data[j].rho,axlim=axlim,scale=scale,zcut=zcut,tcut=tcut,vcut=vcut,focus=focus)
                elif ptype in plotting.pressure_type:        
                    im = plotting.plot_snapshot_fluid(imp.data[j],X,Y,Z,ax=plt.gca(),cmap=cmap,ptype=ptype,plotQ=imp.data[j].P/1e9,axlim=axlim,scale=scale,zcut=zcut,tcut=tcut,vcut=vcut,focus=focus)              
                elif ptype in plotting.entropy_type:        
                    im = plotting.plot_snapshot_fluid(imp.data[j],X,Y,Z,ax=plt.gca(),cmap=cmap,ptype=ptype,plotQ=imp.data[j].S/1e7,axlim=axlim,scale=scale,zcut=zcut,focus=focus,tcut=tcut,vcut=vcut)
                elif ptype in plotting.temperature_type:        
                    im = plotting.plot_snapshot_fluid(imp.data[j],X,Y,Z,ax=plt.gca(),cmap=cmap,ptype=ptype,plotQ=imp.data[j].T,axlim=axlim,scale=scale,zcut=zcut,focus=focus,tcut=tcut,vcut=vcut)
                elif ptype in plotting.phase_type:
                    im = plotting.plot_snapshot_scatter(imp.data[j],cmap=cmap,plotQ=imp.data[j].phase,ptype=ptype,ax=plt.gca(),axlim=axlim,scale=scale,zcut=zcut,tcut=tcut,focus=focus)

                if i>0:
                    plt.gca().set_yticklabels([])
                    plt.gca().set_ylabel('')
                if m != len(imps)*len(types):
                    plt.gca().set_xticklabels([])
                    plt.gca().set_xlabel('')
                if i==0:
                    if ptype in [*plotting.density_type,*plotting.temperature_type]:
                        if uppercaselab:
                            plt.text(0.05,0.95,chr(64+m),ha='left',va='top',fontsize=9,fontweight='bold',color='w',transform=plt.gca().transAxes)
                        else:
                            plt.text(0.05,0.95,chr(96+m),ha='left',va='top',fontsize=9,fontweight='bold',color='w',transform=plt.gca().transAxes)
                    else:
                        if uppercaselab:
                            plt.text(0.05,0.95,chr(64+m),ha='left',va='top',fontsize=9,fontweight='bold',color='k',transform=plt.gca().transAxes)
                        else:
                            plt.text(0.05,0.95,chr(96+m),ha='left',va='top',fontsize=9,fontweight='bold',color='k',transform=plt.gca().transAxes)

                ax = plt.gca()
                if i>0:
                    colpos = ax.get_position()
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
                
                if ptype not in plotting.materials_type:
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
                        plotting.format_colorbar_axis(cbar, ptype, y=False)
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
