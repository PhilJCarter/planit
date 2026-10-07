from .globaldefs import *

import numpy as npy
import scipy
import matplotlib
import matplotlib.pyplot as plt


def plot_snapshot_scatter(snap,ax=None,cmap=plt.get_cmap('plasma'),plotQ=None,ptype='mat',axlim=1.,scf=1.,scale='Earth',zcut=0.,tcut=0.,focus='potmin',rhomin=1e-5):
    #phase flag limits
    phmin=2.5
    phmax=8.5

    if ax is None:
        ax = plt.gca()

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


def plot_snapshot_fluid(snap,X=npy.empty([]),Y=npy.empty([]),Z=npy.empty([]),zi=npy.empty([]),ax=None,cmap=plt.get_cmap('plasma'),plotQ=None,ptype='rho',axlim=1.,scf=1.,scale='Earth',zcut=0.,tcut=0.,vcut=None,rhomin=1e-5,focus='potmin'):
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
    Pmin=1.e-5
    Pmax=1000.
    vi = vit = None

    if ax is None:
        ax = plt.gca()

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
            cols = cmap(cols)
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
