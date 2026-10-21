"""
   planit plotting functions
"""
from .globaldefs import *

import numpy as npy
import scipy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# plot type identifiers
materials_type = ['mat','materials','mats','material']
phase_type = ['phase',]
density_type = ['density','rho']
entropy_type = ['entropy','ent','S']
pressure_type = ['pressure', 'P']
temperature_type = ['temperature', 'T', 'temp']


def get_scale_fac(scale):
    if scale not in ['Mm', 'km', 'earth', 'Earth'] and type(scale) != int and type(scale) != float:
        raise TypeError('scale must be a recognized label or a number')
    if scale == 'Mm':
        scf = 1e8
    elif scale == 'km':
        scf = 1e5
    elif scale == 'earth' or scale == 'Earth':
        scf = Rearth
    else:
        scf = scale
    return scf


def get_axis_units(scale):
    if scale not in ['Mm', 'km', 'earth', 'Earth'] and type(scale) != int and type(scale) != float:
        raise TypeError('scale must be a recognized label or a number')
    if scale == 'Mm':
        u = ' (Mm)'
    elif scale == 'km':
        u = ' (km)'
    elif scale == 'earth' or scale == 'Earth':
        u = r' (R$_\oplus$)'
    else:
        u = ''
    return u


def get_default_colormap(ptype):
    if ptype in materials_type:
        cmap = plt.get_cmap('cmr.bubblegum').copy()
    elif ptype in density_type:
        cmap = plt.get_cmap('cmr.eclipse').copy()
    elif ptype in entropy_type:
        cmap = plt.get_cmap('magma').copy()
    elif ptype in temperature_type:
        cmap = plt.get_cmap('cmr.sunburst').copy()
    elif ptype in pressure_type:
        cmap = plt.get_cmap('inferno').copy()
    elif ptype in phase_type:
        cmap = plt.get_cmap('plasma', 6).copy()
    else:
        cmap = plt.get_cmap('plasma').copy()
    if ptype in density_type:        
        cmap.with_extremes(under='k')
    else:
        cmap.with_extremes(under='w')
    return cmap


def format_colorbar_axis(cbar, ptype, y=False):
    if y:
        axis = cbar.ax.yaxis
    else:
        axis = cbar.ax.xaxis
    if ptype in density_type:
        axis.set_label_text(r'$\rho$ (g$\,$cm$^{-3}$)')
    elif ptype in entropy_type:
        axis.set_label_text(r'$S$ (kJ$\,$K$^{-1}\,$kg$^{-1}$)')
    elif ptype in pressure_type:
        axis.set_label_text(r'$P$ (GPa)')
    elif ptype in temperature_type:
        locs = [100,300,1000,3000,10000,30000]
        cbar.set_ticks(locs)
        axis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:g}'))
        axis.set_label_text(r'$T$ (K)')
    elif ptype in phase_type:
        cbar.set_ticks([3.,4.,5.,6.,7.,8.])
        #cbar.ax.set_xticklabels(['','s','s+l','l','l+v','v','scf'])
        if y:
            cbar.ax.set_yticklabels(['s','s+l','l','l+v','v','scf'])
        else:
            cbar.ax.set_xticklabels(['s','s+l','l','l+v','v','scf'])
        axis.set_label_text(r'Phase')
    if y:
        axis.set_label_position('right')
    else:
        axis.set_label_position('top')


def plot_snapshot_scatter(snap,ax=None,cmap=plt.get_cmap('plasma').copy(),plotQ=None,ptype='mat',axlim=1.,scale='Earth',zcut=None,tcut=0.,focus='potmin',rhomin=1e-5):
    #phase flag limits
    phmin=2.5
    phmax=8.5

    if ptype not in [*materials_type,*phase_type]: 
        raise ValueError('Unknown plot type:', ptype)

    if ax is None:
        ax = plt.gca()

    scf = get_scale_fac(scale)
    
    ti = (snap.header.time)/3600.
    if npy.ndim(ti)>0:
        ti=ti[0]

    if not zcut:
        zcut = axlim / 200. * scf

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
    if ptype in phase_type:
        plotQ = npy.where(plotQ<=6,plotQ-1,plotQ)
        plotQ = npy.where(plotQ<2,6,plotQ)
        cmin = phmin
        cmax = phmax
        norm = matplotlib.colors.Normalize(vmin=cmin,vmax=cmax,clip=False)
    elif ptype in materials_type:
        #cmin = 0
        #cmax = 1
        norm = None
    select = (modz<zcut)*(x<=axlim*scf)*(x>=-axlim*scf)*(y<=axlim*scf)*(y>=-axlim*scf)
    im = ax.scatter(x[select]/scf,y[select]/scf,s=0.1,c=plotQ[select],alpha=1.,cmap=cmap,norm=norm,rasterized=True) #s=0.8

    labcolor='k'
    if ti<1.1:
        ax.text(0.96,0.96,'{:.2f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color=labcolor,transform=plt.gca().transAxes)
    elif ti>23.8:
        ax.text(0.96,0.96,'{:.0f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color=labcolor,transform=plt.gca().transAxes)
    else:
        ax.text(0.96,0.96,'{:.1f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color=labcolor,transform=plt.gca().transAxes)

    ax.set_xlim(-axlim,axlim)
    ax.set_ylim(-axlim,axlim)
    ax.set_aspect('equal')
    ax.minorticks_on()        

    unitstr = get_axis_units(scale)
    ax.set_ylabel('y' + unitstr)
    ax.set_xlabel('x' + unitstr)

    return im


def plot_snapshot_fluid(snap,X=npy.empty([]),Y=npy.empty([]),Z=npy.empty([]),ax=None,cmap=plt.get_cmap('plasma').copy(),plotQ=None,ptype='rho',axlim=1.,scale='Earth',zcut=None,tcut=0.,vcut=None,rhomin=1e-5,focus='potmin'): # zi=npy.empty([])
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

    if ptype not in [*density_type,*entropy_type,*pressure_type,*temperature_type]:
        raise ValueError('Unknown plot type:',ptype)

    if ax is None:
        ax = plt.gca()

    scf = get_scale_fac(scale)

    if not zcut:
        zcut = axlim / 200. * scf

    # number of cells for grid
    Ng = 1201j
    #Ngz = 21j
    #if zi.ndim ==0:
    #    zi=npy.linspace(-axlim/10.,axlim/10.,int(Ngz.imag))
    if X.ndim==0 or Y.ndim==0 or Z.ndim==0:
        #X,Y,Z = npy.mgrid[-axlim:axlim:(Ng),-axlim:axlim:(Ng),-axlim/10.:axlim/10.:(Ngz)]
        X,Y = npy.mgrid[-axlim:axlim:(Ng),-axlim:axlim:(Ng)]
        Z = npy.full_like(X, 0)

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

    #if snap.header.time >= tcut and focus=='potmin':
    #    coz = (z[snap.pot==snap.pot.min()])[0]/scf #s.z[modz<zcut]
    #else:
    coz = 0.0

    if snap.header.time<tcut and vcut:
        if ptype not in density_type:
            vit = scipy.interpolate.griddata((x[(snap.vx>vcut)*(modz<zcut)]/scf,y[(snap.vx>vcut)*(modz<zcut)]/scf,z[(snap.vx>vcut)*(modz<zcut)]/scf),plotQ[(snap.vx>vcut)*(modz<zcut)],(X,Y,Z),method='linear',fill_value=1.e-18)
        rhoit = scipy.interpolate.griddata((x[(snap.vx>vcut)*(modz<zcut)]/scf,y[(snap.vx>vcut)*(modz<zcut)]/scf,z[(snap.vx>vcut)*(modz<zcut)]/scf),snap.rho[(snap.vx>vcut)*(modz<zcut)],(X,Y,Z),method='linear',fill_value=1.e-18)
    else:
        vcut = 2*snap.vel.max()
    if ptype not in density_type:
        vi = scipy.interpolate.griddata((x[(snap.vx<vcut)*(modz<zcut)]/scf,y[(snap.vx<vcut)*(modz<zcut)]/scf,z[(snap.vx<vcut)*(modz<zcut)]/scf),plotQ[(snap.vx<vcut)*(modz<zcut)],(X,Y,Z),method='linear',fill_value=1.e-18)
    rhoi = scipy.interpolate.griddata((x[(snap.vx<vcut)*(modz<zcut)]/scf,y[(snap.vx<vcut)*(modz<zcut)]/scf,z[(snap.vx<vcut)*(modz<zcut)]/scf),snap.rho[(snap.vx<vcut)*(modz<zcut)],(X,Y,Z),method='linear',fill_value=1.e-18)
    if snap.header.time<tcut and rhoit is not None:
        if vi is not None:
            vi = npy.where(vi>vit,vi,vit)
        rhoi = npy.where(rhoi>rhoit,rhoi,rhoit)
    #nn = (npy.nonzero(zi==(zi[zi<=coz])[-1])[0])[0]
    norm = matplotlib.colors.LogNorm(vmin=rhomin,vmax=rhomax,clip=False)
    if vi is not None:
        if ptype in pressure_type:
            cmin = Pmin
            cmax = Pmax
            norm = matplotlib.colors.LogNorm(vmin=cmin,vmax=cmax,clip=False)
            #cols = vi[:,:,nn].T #matplotlib.colors.LogNorm(vmin=cmin,vmax=cmax,clip=True)(vi[:,:,nn].T)
            cols = vi[:,:].T #matplotlib.colors.LogNorm(vmin=cmin,vmax=cmax,clip=True)(vi[:,:,nn].T)
        elif ptype in entropy_type:
            cmin = Smin
            cmax = Smax
            norm = matplotlib.colors.Normalize(vmin=cmin,vmax=cmax,clip=False)
            #cols = matplotlib.colors.Normalize(vmin=cmin,vmax=cmax,clip=True)(vi[:,:,nn].T)
            #alphas = matplotlib.colors.LogNorm(vmin=0.05*rhomin,vmax=rhomax,clip=True)(rhoi[:,:,nn].T)
            cols = matplotlib.colors.Normalize(vmin=cmin,vmax=cmax,clip=True)(vi[:,:].T)
            alphas = matplotlib.colors.LogNorm(vmin=0.05*rhomin,vmax=rhomax,clip=True)(rhoi[:,:].T)
            cols = cmap(cols)
            cols[..., -1] = alphas
        elif ptype in temperature_type:
            cmin = Tmin
            cmax = Tmax
            norm = matplotlib.colors.LogNorm(vmin=cmin,vmax=cmax,clip=False)
            #cols = vi[:,:,nn].T #matplotlib.colors.LogNorm(vmin=cmin,vmax=cmax,clip=True)(vi[:,:,nn].T)
            cols = vi[:,:].T #matplotlib.colors.LogNorm(vmin=cmin,vmax=cmax,clip=True)(vi[:,:,nn].T)
    else:
        #cols = rhoi[:,:,nn].T
        cols = rhoi[:,:].T

    im = ax.imshow(cols,origin='lower',extent=[-axlim,axlim,-axlim,axlim],cmap=cmap,norm=norm)

    if ptype in [*density_type,*temperature_type]:
        ax.tick_params(colors='w',which='both',labelcolor='k')
        ax.spines['top'].set_color('w')
        ax.spines['bottom'].set_color('w')
        ax.spines['left'].set_color('w')
        ax.spines['right'].set_color('w')
        labcolor = 'w'
    else:
        labcolor='k'
    if ti<1.1:
        ax.text(0.96,0.96,'{:.2f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color=labcolor,transform=plt.gca().transAxes)
    elif ti>23.8:
        ax.text(0.96,0.96,'{:.0f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color=labcolor,transform=plt.gca().transAxes)
    else:
        ax.text(0.96,0.96,'{:.1f}'.format(ti),ha='right',va='top',fontsize=9,fontweight='bold',color=labcolor,transform=plt.gca().transAxes)

    ax.set_xlim(-axlim,axlim)
    ax.set_ylim(-axlim,axlim)
    ax.set_aspect('equal')
    ax.minorticks_on()
      
    unitstr = get_axis_units(scale)
    ax.set_ylabel('y' + unitstr)
    ax.set_xlabel('x' + unitstr)

    return im
