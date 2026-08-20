"""
   planit SPH planet generation functions
"""

from .main import *
from .globaldefs import *
from .snaptools import Snapshot
from . import eos
#from . import utils

import numpy as npy
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
import seagen


G_mks = 6.67E-11 # Gravitational constant  m3/kg/s2
Rcmb = 348000000. # CMB radius in cm (from PREM)


class planet_profile:
    """
    1D planet profile class
    
    load()
    write()
    add_isentropic_layer()
    add_adiabatic_layer()
    """
    def __init__(self):
        self.M = 0.
        self.cf = 0.
        self.rarr = npy.array([])
        self.density = npy.array([])
        self.pressure = npy.array([])
        self.temperature = npy.array([])
        self.entropy = npy.array([])
        self.mat = npy.array([])
    
    def write(self, file='profile.dat'):
        npy.savetxt(file,npy.transpose([self.rarr, self.mat, self.entropy, self.density, self.pressure, self.temperature]))
        
    def load(self, file='profile.dat'):
        (self.rarr, self.mat, self.entropy, self.density, self.pressure, self.temperature) = npy.loadtxt(file,unpack=True)

    def add_isentropic_layer(self,mass=0,Pmin=0,iendprevlayer=0,isentrope=None,dR=Rearth/3000.,maxiter=200, firstlayer=False, masstolerance=1e-3):
        if isentrope is None:
            raise ValueError('isentrope must be defined')
        if len(self.rarr) != len(self.density) != len(self.temperature) != len(self.pressure):
            raise ValueError('Arrays rarr, darr, parr, tarr must have same length')
        if len(self.rarr)<1:
            raise ValueError('Central values must be provided!')
    
        #if self.M == 0:
        #    firstlayer = True
        menclosed = self.M
    
        j = 0
        ii = len(self.rarr)
        ri = self.rarr[-1]
        while (self.M-menclosed) < (mass)*(1.-masstolerance) and self.pressure[-1] > Pmin and j < maxiter:
            ri +=dR # m
            self.rarr = npy.append(self.rarr,ri)
            mlayer = 4.*npy.pi*self.rarr[ii]*self.rarr[ii]*dR*self.density[ii-1] # g
            if ii == iendprevlayer+1:
                density0 = npy.interp(self.pressure[ii-1]/1.E10,isentrope.pressure,isentrope.density)
                if self.M > 0:
                    dp = G*self.M*density0*dR/self.rarr[ii]/self.rarr[ii]
                else:
                    dp = G*density0*dR/self.rarr[ii]/self.rarr[ii] # Pa
            else:
                dp = G*self.M*self.density[ii-1]*dR/self.rarr[ii]/self.rarr[ii] # Pa
            self.pressure = npy.append(self.pressure, self.pressure[ii-1]-dp) # 
            if firstlayer:
                di = npy.interp(self.pressure[ii]/1.E10,isentrope.pressure,isentrope.density) # g/cm3
            else:
                di = npy.interp(self.pressure[ii-1]/1.E10,isentrope.pressure,isentrope.density) # g/cm3
            self.density = npy.append(self.density,di) # g/cm3
            if firstlayer:
                ti = npy.interp(self.pressure[ii]/1.E10,isentrope.pressure,isentrope.temperature) # K
            else:
                ti = npy.interp(self.pressure[ii-1]/1.E10,isentrope.pressure,isentrope.temperature) # K
            self.temperature = npy.append(self.temperature,ti) # K
            self.M += mlayer
            ii += 1
            j += 1

    def add_adiabatic_layer(self,mass=0,Pmin=0,iendprevlayer=0,EOS=None,dR=Rearth/3000.,maxiter=200, firstlayer=False, masstolerance=1e-3):
        if EOS is None:
            raise ValueError('EOS must be defined')
        if len(self.rarr) != len(self.density) != len(self.temperature) != len(self.pressure):
            raise ValueError('Arrays rarr, darr, parr, tarr must have same length')
        if len(self.rarr)<1:
            raise ValueError('Central values must be provided!')

        if EOS.TYPE != 'HM80':
            S = eos.tabinterp.from_rhoT('S',self.density[-1],self.temperature[-1],EOS.make_passer_class())
            isentrope = eos.isentrope_class(S,EOS.MODELNAME)
            self.add_isentropic_layer(mass=mass,Pmin=Pmin,iendprevlayer=iendprevlayer,isentrope=isentrope,dR=dR,maxiter=maxiter,masstolerance=masstolerance)
        else:  # HM80 only
            #if self.M == 0:
            #    firstlayer = True
            menclosed = self.M

            j = 0
            ii = len(self.rarr)
            ri = self.rarr[-1]
            while (self.M-menclosed) < (mass)*(1.-masstolerance) and self.pressure[-1] > Pmin and j < maxiter:
                ri +=dR # m
                self.rarr = npy.append(self.rarr,ri)
                mlayer = 4.*npy.pi*self.rarr[ii]*self.rarr[ii]*dR*self.density[ii-1] # g
                if ii == iendprevlayer+1:
                    density0 = HM80_rho_P_T(self.pressure[ii-1],self.temperature[ii-1],rho_est=0.9*self.density[ii-1])
                    if self.M > 0:
                        dp = G*self.M*density0*dR/self.rarr[ii]/self.rarr[ii]
                    else:
                        dp = G*density0*dR/self.rarr[ii]/self.rarr[ii] # Pa
                else:
                    dp = G*self.M*self.density[ii-1]*dR/self.rarr[ii]/self.rarr[ii] # Pa
                self.pressure = npy.append(self.pressure, self.pressure[ii-1]-dp) # 
                #U = HM80_U_rho_T(self.density[ii-1],self.temperature[ii-1])
                #P = HM80_P_rho_T(rho,T) #tabinterp.from_rhoU1D('P',rho,U,EOS)
                newrho = HM80_rho_P_T(self.pressure[ii],self.temperature[ii-1],rho_est=self.density[ii-1])
                self.density = npy.append(self.density,newrho)
                if ii == iendprevlayer+1:
                    newT = HM80_adiabat_T(density0,self.density[ii],self.temperature[ii-1])
                else:
                    newT = HM80_adiabat_T(self.density[ii-1],self.density[ii],self.temperature[ii-1])
                self.temperature = npy.append(self.temperature,newT)
                self.M += mlayer
                ii+=1
                j+=1
                if j%200 == 0:
                    print(newrho,newT,self.pressure[-1])



def HM80_U_rho_T(rho,T):
    return HM80_Cv_rho_T(rho,T)*T

def HM80_Cv_rho_T(rho,T):
    c1 = 2.3638
    c2 = -4.9842e-5
    c3 = 1.1788e-8
    c4 = -3.8101e-4
    c5 = 2.6182
    c6 = 0.45053
    
    FMW = 2.2857143
    
    return 8.31446e7/FMW * ( c1 + c2*T + c3*T**2 + c4*rho*T + c5*rho + c6*rho**2 )

def HM80_P_rho_T(rho,T):
    u1 = -16.05895
    u2 = 1.22808
    u3 = -0.0217930
    u4 = 0.141021
    u5 = 0.147156
    u6 = 0.277708
    u7 = 0.0455347
    u8 = -0.0558596
    
    rho0 = 0.005
    
    y = npy.log(T)
    x = npy.log(rho/rho0)
    
    lnP = u1 + u2*y + u3*y**2 + u4*x*y +u5*x + u6*x**2 +u7*x**3 +u8*y*x**2 #Mbar

    return npy.exp(lnP)*1e6*1e6

def HM80_gamma_rho_T(rho,T):
    rho0 = 0.005
    y = npy.log(T)
    x = npy.log(rho/rho0)
    return HM80_gamma_lnrho_lnT(x,y)

def HM80_gamma_lnrho_lnT(x,y):
    b1 = 0.328471
    b2 = 0.0286529
    b3 = -0.00139609
    b4 = -0.0232258
    b5 = 0.0579055
    b6 = 0.0454488
    return b1 + b2*y + b3*y**2 + b4*x*y + b5*x + b6*x**2

def HM80_adiabat_T(rho0,rho1,T0):
    import scipy
    rhoref = 0.005
    sol = scipy.integrate.solve_ivp(HM80_gamma_lnrho_lnT,[npy.log(rho0/rhoref),npy.log(rho1/rhoref)],[npy.log(T0)])
    return npy.exp(sol.y[0][-1])

def HM80_rho_P_T(P, T, rho_est=None, tolerance=1e-3):

    maxiter = 5000
    
    eos_rho_min = 0.005
    eos_rho_max = 1000.

    if rho_est:
        rho_min = 0.5 * rho_est
        rho_max = 2. * rho_est
        rho = rho_est
    else:
        rho_min = 0.005
        rho_max = 1000.
        rho = 1.
    #log_rho_mid = (npy.log10(rho_max)-npy.log10(rho_min))/2. + npy.log10(rho_min)

    #rho_min = max(rho_min,eos_rho_min)
    #rho_max = min(rho_max,eos_rho_max)

    derivedP = HM80_P_rho_T(rho,T)
    T_low = HM80_adiabat_T(rho_min,rho_min,T)
    derivedP_low = HM80_P_rho_T(rho_min,T_low)
    T_mid = HM80_adiabat_T(rho_min,rho,T)
    derivedP_mid = HM80_P_rho_T(rho,T_mid)
    T_high = HM80_adiabat_T(rho_min,rho_max,T)
    derivedP_high = HM80_P_rho_T(rho_max,T_high)

    if derivedP_low < derivedP_high < P:
        return HM80_rho_P_T(P, T, rho_est=rho_max)
    elif derivedP_high > derivedP_low > P:
        return HM80_rho_P_T(P, T, rho_est=rho_min)
    
    j=0
    
    while npy.abs(P-derivedP)/P > tolerance and j < maxiter:
        T_low = HM80_adiabat_T(rho_min,rho_min,T)
        derivedP_low = HM80_P_rho_T(rho_min,T_low)
        T_mid = HM80_adiabat_T(rho_min,rho,T)
        derivedP_mid = HM80_P_rho_T(rho,T_mid)
        T_high = HM80_adiabat_T(rho_min,rho_max,T)
        derivedP_high = HM80_P_rho_T(rho_max,T_high)
        
        if derivedP_low > P:
            print('WARNING: P:', P,'could not be matched, low:',derivedP_low,derivedP_mid)
            #return rho_min
        elif derivedP_high < P:
            print('WARNING: P:', P,'could not be matched, high:',derivedP_high,derivedP_mid)
            #return rho_max
        if derivedP_mid > P:
            rho_max = rho
        else:
            rho_min = rho
            derivedP_low = derivedP_mid
            T_low = T_mid
        derivedP = derivedP_low
        T = T_low
        rho = npy.power(10,(npy.log10(rho_max)-npy.log10(rho_min))/2. + npy.log10(rho_min))
        #rho = (rho_max-rho_min)/2.
        #print(j,rho,T,derivedP)
        j += 1
    
    #print(P,derivedP,rho,rho_min,rho_max,j,T,T_high)   

    return rho
    


def planet_density(m):
    """
    Estimate density from planet mass.
    Based on Müller et al., A&A 686, A296 (2024).
    """
    if m < 3.8:
        d = 5.12 * m**0.12
    elif m < 183:
        d = 16.6 * m**-0.76
    else:
        d = 4.39e-4 * m**1.26
    return d
    


def make_1D_planet(mass=Mearth, corefraction=0.325, Pmin=1.e6, Score=1.81, Smantle=3.02,
        mtolerance=1e-3, layer1='iron', layer2='forsterite', layers=[], S=[], mantlepotT=False,
        plot=False, fixcoreT=False, rhocent=None, verbose=False):
    """
    Create a 1D planet profile
    
    mass - list of layer masses (inside to out) or total mass of 2 layer planet
    corefraction - core mass fraction for 2 layer planet
    Pmin - surface pressure (Pa)
    Score - core entropy for 2 layer planet
    Smantle - 2nd layer entropy for 2 layer planet
    mtolerance - tolerance for total mass of planet
    layer1 - 1st layer material for 2 layer planet
    layer 2 - 2nd layer material for 2 layer planet
    layers - list of materials for layers (inside to out)
    S - list of layer entropies (inside to out)
    mantlepotT - mantle potential temperature for 2 layer planet
    plot - plot pressure-temperature profile?
    fixcoreT - ensure outer core is at least as hot as inner mantle (2nd layer)?
    rhocent - initial central density to use (optional, use for troublesome cases)
    verbose - print extra information
    
    """
    if len(layers) != len(S) != len(mass):
        raise ValueError('number of layers must match. layers, S, mass:', len(layers), len(S), len(mass))
    if not layers or len(layers) == 2:
        if layers:
            layer1 = layers[0]
            layer2 = layers[1]
            Score = S[0]
            Smantle = S[1]
            totmass = sum(mass)
            corefraction = mass[0]/totmass
            mass = totmass
        return make_1D_2L_planet(mass=mass,corefraction=corefraction,Pmin=Pmin,Score=Score,Smantle=Smantle,mtolerance=mtolerance,layer1=layer1,layer2=layer2,mantlepotT=mantlepotT,plot=plot,fixcoreT=fixcoreT,verbose=verbose,rhocent=rhocent)
    else:
        return make_1D_NL_planet(mass=mass, Pmin=Pmin, S=S, mtolerance=mtolerance, layers=layers, 
            plot=plot, fixcoreT=fixcoreT, verbose=verbose, rhocent=rhocent, mantlepotT=mantlepotT)



def make_1D_2L_planet(mass=Mearth, corefraction=0.325, Pmin=1.e6, Score=1.81, Smantle=3.02,
        mtolerance=1e-3, layer1='iron', layer2='forsterite', mantlepotT=False, plot=False,
        fixcoreT=False, verbose=False, rhocent=None):
    
    mcore = corefraction * mass
    mmantle = (1-corefraction) * mass

    mtotal = mcore + mmantle

    Pcenter = 0
    
    ly1EOS = eos.select(layer1)
    if ly1EOS is None:
        print('Unknown core EOS')
        return
    
    ly2EOS = eos.select(layer2)
    if ly2EOS is None:
        print('Unknown mantle EOS')
        return
    
    if mantlepotT:
        refP = Pmin
        refP_T = npy.zeros(ly2EOS.NT)
        refP_S = npy.zeros(ly2EOS.NT)
        refP_rho = npy.zeros(ly2EOS.NT)
        #it0 = npy.where(ly2EOS.T >= ly2EOS.T0REF)[0]
        id0 = npy.arange(ly2EOS.ND)
        for iit in range(0,ly2EOS.NT):
            refP_T[iit] = ly2EOS.T[iit]
            refP_S[iit] = npy.interp(refP/1.E10,ly2EOS.P[iit,id0],ly2EOS.S[iit,id0])
            refP_rho[iit] = npy.interp(refP/1.E10,ly2EOS.P[iit,id0],ly2EOS.rho[id0])
        Smantle = npy.interp(mantlepotT,refP_T,refP_S)*1e3
    
    # first extract the isentropes for the planet from the EOS tables
    mantle = eos.isentrope_class(Smantle,layer2)
    
    Tcmb_accept = False
    
    # overall loop to allow core temperature adjustment (no max iterations!)
    while not Tcmb_accept:
        
        core = eos.isentrope_class(Score,layer1)

        if plot:
            PREM = PREMclass()
            fig = plt.figure(figsize=(7,5))
            plt.plot(PREM.pressure,PREM.temperature,'-.',color='xkcd:deep blue',label='PREM profile')
            plt.plot(mantle.pressure,mantle.temperature,'-',color='xkcd:purple',label='mantle isentrope',markersize=10)
            plt.plot(core.pressure,core.temperature,'-',color='xkcd:tangerine',label='core isentrope',markersize=10)

            plt.plot(ly2EOS.mc.Pl,ly2EOS.mc.T,'-',color='black',label='forsterite MC',markersize=10)
            plt.plot(ly1EOS.mc.Pl,ly1EOS.mc.T,'--',color='black',label='iron MC',markersize=10)

            plt.ylim(0.,max(PREM.temperature))
            plt.xlim(0,max(PREM.pressure))
            plt.xlabel('Pressure (GPa)')
            plt.ylabel('Temperature (K)')
            plt.legend()
            plt.show()

        if rhocent:
            rhocenter = rhocent
        else:
            rhocenter = 2*planet_density(mtotal/Mearth)
        changefac = 0.00003*rhocenter**4.2

        if verbose:
            print('m:',mtotal/Mearth,'fac:',changefac,' rho:',rhocenter )

        r_est = (mtotal/(4./3.*npy.pi*rhocenter/2.))**(1./3.)
        dR = npy.floor(r_est/1e5)*1e5 / 3000
        # ensure dR is not too small
        if dR < 0.025e5:
            dR = 0.025e5

        Pi = 0.
        rhoi = rhocenter

        darr = npy.full(1,rhoi)
        parr = npy.full(1,Pi)
        rarr = npy.zeros(1)

        planet = planet_profile()
        planet.rarr = rarr
        planet.density = darr
        planet.pressure = parr
        
        itercount = 0
        maxiter = 5000
        maxiterm = 5000

        if mtolerance < 5e-4:
            changefac *= 0.2
            maxiterm *= 3
            dR *= 0.3
        elif mtolerance < 1e-3:
            changefac *= 0.4
            maxiterm *= 1.2
            dR *= 0.8

        if verbose:
            print('dR: ', dR/1e5, 'km')


        # outer loop to find the correct total mass
        while npy.abs(mtotal-planet.M)/mtotal > mtolerance and itercount < maxiter:
            if (itercount%100 == 0 and verbose) or (itercount%10000 == 0 and itercount!=0):
                print(planet.pressure[0],planet.density[0],planet.M/mtotal)

            rarr = npy.zeros(1)
            darr = npy.full(1,rhoi) # Pa
            parr = npy.full(1,npy.interp(rhoi,core.density,core.pressure)*1.E10) # 
            tarr = npy.full(1,npy.interp(rhoi,core.density,core.temperature)) # K

            #ri = rarr[0]
            #ii=1
            #coreiter=0
            
            planet.M = 0.
            planet.rarr = rarr
            planet.density = darr
            planet.pressure = parr
            planet.temperature = tarr
            
            # inner loop 1 to make core of required mass
            planet.add_isentropic_layer(mass=mcore,Pmin=Pmin,iendprevlayer=0,isentrope=core,dR=dR,maxiter=maxiterm,masstolerance=mtolerance)

            iendcore = len(planet.rarr)-1
            
            # inner loop 2 to add mantle up to surface pressure
            planet.add_isentropic_layer(mass=mtotal-mcore,Pmin=Pmin,iendprevlayer=iendcore,isentrope=mantle,dR=dR,maxiter=maxiterm,masstolerance=mtolerance)   

            # adjust initial density if total mass wrong
            if planet.M < mtotal:
                #rhoi *= 1.0+2*mtolerance*changefac
                #Pi *= 1.0+2*mtolerance*changefac
                rhoi *= 1.0 + (0.04*(mtotal-planet.M)/mtotal + 1.8*mtolerance)*changefac
                #Pi *= 1.0+2*mtolerance*changefac
            elif planet.M > mtotal:
                #rhoi *= 1.0-1*mtolerance*changefac
                #Pi *= 1.0-1*mtolerance*changefac
                rhoi *= 1.0 - (0.02*(planet.M-mtotal)/mtotal + 0.5*mtolerance)*changefac
                #Pi *= 1.0-1*mtolerance*changefac

            itercount += 1
            
        if not fixcoreT:
            Tcmb_accept = True
    
        if planet.temperature[iendcore] < npy.interp(planet.pressure[iendcore]/1.E10,mantle.pressure,mantle.temperature):
            print('WARNING! Core colder than mantle!   {:10.3f}  {:10.3f}  | S: {:.3f}'.format(planet.temperature[iendcore],npy.interp(planet.pressure[iendcore]/1.E10,mantle.pressure,mantle.temperature),Score))
            if fixcoreT:
                Score += 0.01
        else:
            Tcmb_accept = True
    
    print('Iterations:    ',itercount,'\n')        

    planet.cf = mcore/planet.M
    planet.mat = npy.where(npy.arange(len(planet.rarr))<=iendcore,0,1)
    planet.entropy = npy.where(npy.arange(len(planet.rarr))<=iendcore,Score,Smantle)*1e7
    
    print('MODEL PLANET:')
    print('Score (kJ/K/kg)         = {:10.3f}'.format(Score))
    print('Smantle (kJ/K/kg)       = {:10.3f}'.format(Smantle))
    print('Tcmb (K)                = {:10.3f}  [{:10.3f}]'.format(planet.temperature[iendcore],npy.interp(planet.pressure[iendcore]/1.E10,mantle.pressure,mantle.temperature)))
    print('Mantle Tp (K)           = {:10.3f}'.format(planet.temperature[-1])) #min(tarr)
    print('Pcenter (GPa)           = {:10.3f}'.format(planet.pressure[0]/1.E10))
    print('Tcenter (K)             = {:10.3f}'.format(planet.temperature[0]))
    print('rho center (g/cm3)      = {:10.3f}'.format(planet.density[0]))
    print('Core radius (km, Rcmb)  = {:10.3f}{:10.3f}'.format(planet.rarr[iendcore]/1.e5, planet.rarr[iendcore]/Rcmb))
    print('CMB pressure (GPa)      = {:10.3f}'.format(planet.pressure[iendcore]/1.E10))
    print('Radius (km, Rearth)     = {:10.3f}{:10.3f}'.format(planet.rarr[-1]/1.e5, planet.rarr[-1]/Rearth))
    print('Surface pressure (GPa)  = {:10.3f}'.format(planet.pressure[-1]/1.E10))
    print('Surface gravity (m/s2)  = {:10.3f}'.format(G*planet.M/planet.rarr[-1]/planet.rarr[-1]/100.))
    print('Vesc (km/s)             = {:10.3f}'.format(npy.sqrt(2*G*planet.M/planet.rarr[-1])/1.E5))
    print('Mass/Mearth, Mcore/Mass = {:10.5f}{:10.4f}'.format(planet.M/Mearth, mcore/planet.M))
    
    return planet,core,mantle


def make_1D_NL_planet(mass=[0.3*Mearth,0.7*Mearth], Pmin=1.e6, S=[1.81,3.02],
        mtolerance=1e-3, layers=['iron','forsterite'], mantlepotT=False, plot=False,
        fixcoreT=False, verbose=False, rhocent=None):
    
    mtotal = sum(mass)

    Pcenter = 0
    
    layerEOS = []
    isentropes = []
    for layer,layerS in zip(layers,S):
        EOS = eos.select(layer)
        if EOS is None:
            raise ValueError('Unknown EOS:',layer)
        layerEOS.append(EOS)
        if EOS.TYPE == 'HM80' or layerS == 'adiabat':
            isentropes.append('adiabat')
        else:
            isentropes.append(eos.isentrope_class(layerS,layer))    

    Tcmb_accept = False
    
    # overall loop to allow core temperature adjustment (no max iterations!)
    while not Tcmb_accept:
        
        isentropes[0] = eos.isentrope_class(S[0],layers[0])

        if plot:
            PREM = PREMclass()
            fig = plt.figure(figsize=(7,5))
            plt.plot(PREM.pressure,PREM.temperature,'-.',color='xkcd:deep blue',label='PREM profile')
            for isentrope in isentropes:
                plt.plot(isentrope.pressure,isentrope.temperature,label=str(isentrope.material)+' isentrope',markersize=10)
            
            for EOS in layerEOS:
                if EOS.TYPE == 'ANEOS':
                    plt.plot(EOS.mc.Pl,EOS.mc.T,'--',color='black',label=EOS.MODELNAME+' MC',markersize=10)

            plt.ylim(0.,max(PREM.temperature))
            plt.xlim(0,max(PREM.pressure))
            plt.xlabel('Pressure (GPa)')
            plt.ylabel('Temperature (K)')
            plt.legend()
            plt.show()

        if rhocent:
            rhocenter = rhocent
        else:
            rhocenter = 2*planet_density(mtotal/Mearth)
        changefac = 0.00003*rhocenter**4.2

        if verbose:
            print('m:',mtotal/Mearth,'fac:',changefac,' rho:',rhocenter )

        r_est = (mtotal/(4./3.*npy.pi*rhocenter/2.))**(1./3.)
        dR = npy.floor(r_est/1e5)*1e5 / 3000
        # ensure dR is not too small
        if dR < 0.025e5:
            dR = 0.025e5

        Pi = 0.
        rhoi = rhocenter

        darr = npy.full(1,rhoi)
        parr = npy.full(1,Pi)
        rarr = npy.zeros(1)

        planet = planet_profile()
        planet.rarr = rarr
        planet.density = darr
        planet.pressure = parr
        
        itercount = 0
        maxiter = 1000
        maxiterm = 5000

        if mtolerance < 5e-4:
            changefac *= 0.2
            maxiterm *= 3
            dR *= 0.3
        elif mtolerance < 1e-3:
            changefac *= 0.4
            maxiterm *= 1.2
            dR *= 0.8

        if verbose:
            print('dR: ', dR/1e5, 'km')


        # outer loop to find the correct total mass
        while npy.abs(mtotal-planet.M)/mtotal > mtolerance and itercount < maxiter:
            if (itercount%100 == 0 and verbose) or (itercount%10000 == 0 and itercount!=0):
                print(planet.pressure[0],planet.density[0],planet.M/mtotal)

            rarr = npy.zeros(1)
            darr = npy.full(1,rhoi) # Pa
            parr = npy.full(1,npy.interp(rhoi,isentropes[0].density,isentropes[0].pressure)*1.E10) # 
            tarr = npy.full(1,npy.interp(rhoi,isentropes[0].density,isentropes[0].temperature)) # K

            planet.M = 0.
            planet.rarr = rarr
            planet.density = darr
            planet.pressure = parr
            planet.temperature = tarr
            
            layerfinalindices = []
            layermasses = []
            # make layers of required masses
            for layermass, layerisentrope, EOS in zip(mass, isentropes, layerEOS):
                if layerisentrope == 'adiabat':
                    planet.add_adiabatic_layer(mass=layermass,Pmin=Pmin,iendprevlayer=len(planet.rarr)-1,EOS=EOS,dR=dR,maxiter=maxiterm,masstolerance=mtolerance)
                else:
                    planet.add_isentropic_layer(mass=layermass,Pmin=Pmin,iendprevlayer=len(planet.rarr)-1,isentrope=layerisentrope,dR=dR,maxiter=maxiterm,masstolerance=mtolerance)
                layerfinalindices.append(len(planet.rarr)-1)
                layermasses.append(planet.M-sum(layermasses))
            
            # adjust initial density if total mass wrong
            if planet.M < mtotal:
                #rhoi *= 1.0+2*mtolerance*changefac
                #Pi *= 1.0+2*mtolerance*changefac
                rhoi *= 1.0 + (0.04*(mtotal-planet.M)/mtotal + 1.8*mtolerance)*changefac
                #Pi *= 1.0+2*mtolerance*changefac
            elif planet.M > mtotal:
                #rhoi *= 1.0-1*mtolerance*changefac
                #Pi *= 1.0-1*mtolerance*changefac
                rhoi *= 1.0 - (0.02*(planet.M-mtotal)/mtotal + 0.5*mtolerance)*changefac
                #Pi *= 1.0-1*mtolerance*changefac

            itercount += 1
            
        if not fixcoreT:
            Tcmb_accept = True
    
        if planet.temperature[layerfinalindices[0]] < npy.interp(planet.pressure[layerfinalindices[0]]/1.E10,isentropes[1].pressure,isentropes[1].temperature):
            print('WARNING! Core colder than mantle!   {:10.3f}  {:10.3f}  | S: {:.3f}'.format(planet.temperature[layerfinalindices[0]],npy.interp(planet.pressure[layerfinalindices[0]]/1.E10,isentropes[1].pressure,isentropes[1].temperature),S[0]))
            if fixcoreT:
                S[0] += 0.01
        else:
            Tcmb_accept = True
    
    print('Iterations:    ',itercount,'\n')        

    planet.cf = layermasses[0]/planet.M
    planet.mat = npy.digitize(range(len(planet.rarr)),layerfinalindices,right=True)
#    planet.entropy = npy.where(npy.arange(len(planet.rarr))<=iendcore,Score,Smantle)*1e7
    
    print('MODEL PLANET:')
    print('Score (kJ/K/kg)         = {:10.3f}'.format(S[0]))
    print('Smantle (kJ/K/kg)       = {:10.3f}'.format(S[1]))
    print('Tcmb (K)                = {:10.3f}  [{:10.3f}]'.format(planet.temperature[layerfinalindices[0]],npy.interp(planet.pressure[layerfinalindices[0]]/1.E10,isentropes[1].pressure,isentropes[1].temperature)))
    print('Mantle Tp (K)           = {:10.3f}'.format(planet.temperature[-1])) #min(tarr)
    print('Pcenter (GPa)           = {:10.3f}'.format(planet.pressure[0]/1.E10))
    print('Tcenter (K)             = {:10.3f}'.format(planet.temperature[0]))
    print('rho center (g/cm3)      = {:10.3f}'.format(planet.density[0]))
    print('Core radius (km, Rcmb)  = {:10.3f}{:10.3f}'.format(planet.rarr[layerfinalindices[0]]/1.e5, planet.rarr[layerfinalindices[0]]/Rcmb))
    print('CMB pressure (GPa)      = {:10.3f}'.format(planet.pressure[layerfinalindices[0]]/1.E10))
    print('Radius (km, Rearth)     = {:10.3f}{:10.3f}'.format(planet.rarr[-1]/1.e5, planet.rarr[-1]/Rearth))
    print('Surface pressure (GPa)  = {:10.3f}'.format(planet.pressure[-1]/1.E10))
    print('Surface gravity (m/s2)  = {:10.3f}'.format(G*planet.M/planet.rarr[-1]/planet.rarr[-1]/100.))
    print('Vesc (km/s)             = {:10.3f}'.format(npy.sqrt(2*G*planet.M/planet.rarr[-1])/1.E5))
    print('Mass/Mearth, Mcore/Mass = {:10.5f}{:10.4f}'.format(planet.M/Mearth, layermasses[0]/planet.M))
    
    return planet,isentropes



def make_SPH_planet(mass=Mearth, corefraction=0.3, Pmin=1.e6, Score=1.81, Smantle=3.03, 
        mtolerance=1e-3, layer1='alloy', layer2='forsterite', layers=[], S=[], mantlepotT=False, 
        plot=False, resolution=5e5, fixcoreT=False, rhocent=None, verbose=False, profile=None):
    """
    mass - list of layer masses (inside to out) or total mass of 2 layer planet
    corefraction - core mass fraction for 2 layer planet
    resolution - number of particles per Earth mass
    Pmin - surface pressure (Pa)
    Score - core entropy for 2 layer planet
    Smantle - 2nd layer entropy for 2 layer planet
    mtolerance - tolerance for total mass of planet
    layer1 - 1st layer material for 2 layer planet
    layer2 - 2nd layer material for 2 layer planet
    layers - list of materials for layers (inside to out)
    S - list of layer entropies (inside to out)
    mantlepotT - mantle potential temperature for 2 layer planet
    plot - plot pressure-temperature profile and x-y slice?
    fixcoreT - ensure outer core is at least as hot as inner mantle (2nd layer)?
    rhocent - initial central density to use (optional, use for troublesome cases)
    verbose - print extra information
    profile - 1D profile from which to make planet (if not supplied 1D prof will be generated)
    """
    
    if profile:
        planet = profile
    else:
        if len(layers) != len(S):
            raise ValueError('Number of layers must match number of entropies:', len(layers), len(S))
        if not layers: # or len(layers) == 2:
            #layer1 = layers[0]
            #layer2 = layers[1]
            #Score = S[0]
            #Smantle = S[1]
            #totmass = sum(mass)
            #corefraction = mass[0]/totmass
            #mass = totmass
            planet,core,mantle = make_1D_planet(plot=plot, mantlepotT=mantlepotT, layer1=layer1, layer2=layer2, mass=mass, corefraction=corefraction, Pmin=Pmin, Score=Score, Smantle=Smantle, mtolerance=mtolerance, fixcoreT=fixcoreT, verbose=verbose, layers=layers, S=S, rhocent=rhocent)
            
        else:
            planet,isentropes = make_1D_planet(plot=plot, mantlepotT=mantlepotT, layer1=layer1, layer2=layer2, mass=mass, corefraction=corefraction, Pmin=Pmin, Score=Score, Smantle=Smantle, mtolerance=mtolerance, fixcoreT=fixcoreT, verbose=verbose, layers=layers, S=S, rhocent=rhocent)
            
        #if len(layers) == 0:
        #    core, mantle = isentropes
    
    partmass = Mearth / resolution   # mass per particle
    Np = int(planet.M / partmass)    # desired total number of particles
    
    if verbose:
        print(partmass,Np)
        print(planet.rarr)
    
    # use seagen to generate spherical planet particles
    particleplanet = seagen.GenSphere(Np,planet.rarr[1:],planet.density[1:],A1_T_prof=planet.temperature[1:],A1_P_prof=planet.pressure[1:],A1_mat_prof=planet.mat[1:],verbosity=0, A1_m_rel_prof=1.0*npy.ones(len(planet.mat[1:])))#, A1_force_more_shells=[False,True])
    if len(layers) == 0:
        S = [Score,Smantle]
    setattr(particleplanet, "S", npy.array(S)[particleplanet.mat]*1e7)
    
    if verbose:
        print(npy.unique(particleplanet.A1_m))
        print( (particleplanet.A1_m.max()-npy.mean(particleplanet.A1_m))/npy.mean(particleplanet.A1_m), (particleplanet.A1_m.min()-npy.mean(particleplanet.A1_m))/npy.mean(particleplanet.A1_m) )

    if plot:
        zcut = 50e5
        plt.scatter(particleplanet.A1_x[npy.abs(particleplanet.z)<zcut]/1e5,particleplanet.A1_y[npy.abs(particleplanet.z)<zcut]/1e5,c=particleplanet.rho[npy.abs(particleplanet.z)<zcut])
        plt.axis('equal')
        plt.colorbar()
        plt.show()

    print('N:', particleplanet.N_picle)
    massresid = partmass-npy.unique(particleplanet.A1_m)
    if verbose:
        print('pm:', massresid.min()/partmass, massresid.max()/partmass)

    # load particle planet into planit snapshot
    sn = Snapshot()
    sn.ic_from_seagen(particleplanet)
    if len(layers) == 0:
        sn.ensure_matIDs((layer1,layer2))
    else:
        sn.ensure_matIDs(layers)
        
    if verbose:
        print(sn.m.sum()/Mearth, sn.m.std()/Mearth)

    sn.m = npy.ones(len(sn.m))*partmass   # set all particle masses to calculated value

    if verbose:
        print(sn.m.sum()/Mearth, sn.m.std()/Mearth)
    
    if not profile:
        if len(layers) == 0:
            return planet, core, mantle, sn, particleplanet
        else:
            return planet, isentropes, sn, particleplanet
    else:
        return sn, particleplanet



# <b>PREM</b>
# PREM is a one-dimensional average structure model for the Earth developed from seismic data. The original reference is:<br>
# <b>Dziewonski, A. M., and D. L. Anderson. 1981. "Preliminary reference Earth model." Phys. Earth Plan. Int. 25:297-356.</b>
# 
# The 1D model has been updated by Panning and Romanowicz (2006). Download the Modified PREM model from the <a href="http://ds.iris.edu/spud/earthmodel/9785674">IRIS web site</a>.
#Durek, J. J., and G. Ekstrom (1996) Modified PREM (Preliminary Reference Earth Model), doi:10.17611/DP/9785674, http://ds.iris.edu/spud/earthmodel/9785674.

class PREMclass:
    """Class to hold PREM data and other 1-D Earth variables."""  # this is a documentation string for this class
    def __init__(self): # self is the default name of the object for internal referencing of the variables in the class
        """A function to initialize the class object.""" # this is a documentation string for this function
        self.NR = 0 # number of radius points
        self.radius = npy.zeros(self.NR) 
        self.density = npy.zeros(self.NR)   
        self.pwavevel = npy.zeros(self.NR)   
        self.swavevel = npy.zeros(self.NR)
        self.pressure = npy.zeros(self.NR)
        self.temperature = npy.zeros(self.NR)
        # not going to use all the variables in the file
        self.units = '' # I like to keep a text note in a structure about the units
        
        self.load()

    def load(self,PREM_filename=datadir+'PREM500_IDV.csv'):
        # Read in PREM: Preliminary Earth Reference Model

        self.radius = npy.loadtxt(PREM_filename,delimiter=',',skiprows=2,usecols=[0]) # radius in m
        self.density = npy.loadtxt(PREM_filename,delimiter=',',skiprows=2,usecols=[1]) # density in kg/m3
        self.pwavevel = npy.loadtxt(PREM_filename,delimiter=',',skiprows=2,usecols=[2]) # p-wave velocity m/s
        self.swavevel = npy.loadtxt(PREM_filename,delimiter=',',skiprows=2,usecols=[3]) # s-wave velocity m/s
        self.NR = len(self.radius) # number of radius points
        self.units = 'radius (m), density (kg/m3), pwavevel (m/s), swavevel (m/s)'
        
        # start at the surface and integrate via a for loop to the center of the planet
        
        # calculate the thickness of each layer in the PREM model using the roll function
        PREM_dr = npy.roll(self.radius,-1)-self.radius 
        PREM_dr[self.NR-1] = 0. # we are not using the last entry in the list because there are NR-1 layers
        #print(PREM_dr)
        
        # calculate the mass of each layer
        # density x area x thickness of each layer
        PREM_mass_rad = self.density*(4.*npy.pi*self.radius*self.radius*PREM_dr) 
        
        # core radius
        ir0 = npy.where(self.density < 6000.)[0]
        PREM_rcore = self.radius[ir0[0]]
                
        self.pressure = npy.zeros(self.NR) # make array of zeros for pressure of the same length as the arrays in the PREM model
        # The first entry is the middle of the planet, so start at the surface and integrate inwards
        for i in range(self.NR-2,0,-1):
            self.pressure[i] = self.pressure[i+1]+G_mks*npy.sum(PREM_mass_rad[0:i-1])*self.density[i]*PREM_dr[i]/self.radius[i]/self.radius[i]
        
        PREM_mass_enclosed = npy.zeros(self.NR) # make a new array
        for i in range(1,self.NR):
            PREM_mass_enclosed[i] = npy.sum(PREM_mass_rad[0:i])
        
        # use SESAME units
        self.pressure = self.pressure/1.E9 # GPa
        
        Anzellini_filename = datadir+'Anzellini-2013-Science-coretemp.csv'
        Anzellini_pressure  = npy.loadtxt(Anzellini_filename,delimiter=',',skiprows=0,usecols=[0]) # GPa
        Anzellini_temperature  = npy.loadtxt(Anzellini_filename,delimiter=',',skiprows=0,usecols=[1]) # K
        # we want to use this data with PREM but it doesn't have the same number of points
        # interpolate the Anzellini profile to match the PREM profile
        # the following defines a function = y(x)
        interp_func = interpolate.interp1d(Anzellini_pressure,Anzellini_temperature,fill_value="extrapolate") 
        self.temperature = interp_func(self.pressure)



def plot_planet_prof(planet,particleplanet=None,show=False,path=False,coreEOS='iron',mantleEOS='Forsterite'):
    if particleplanet:
        inclpart = True
        if not hasattr(particleplanet, 'r'):
            particleplanet.r = npy.sqrt(particleplanet.x**2 + particleplanet.y**2 + particleplanet.z**2)
        
    if planet.M >= 0.1*Mearth:
        showPREM = True
        PREM = PREMclass()
    else:
        showPREM = False

    cEOS = eos.select(coreEOS)
    if cEOS is None:
        print('Unknown core EOS')
        return
    mEOS = eos.select(mantleEOS)
    if mEOS is None:
        print('Unknown core EOS')
        return
        
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8,12))
    plt.subplots_adjust(wspace=0.25)
    #------------------------------
    ai=0
    aj=0
    axes[ai,aj].set_xlabel('Radius (km)')
    axes[ai,aj].set_ylabel('Density (g/cm$^3$)')
    if inclpart:
        axes[ai,aj].scatter(particleplanet.r/1e5,particleplanet.rho,s=6,c=particleplanet.S/1e7,vmax=4.)
    axes[ai,aj].plot(planet.rarr/1.e5,planet.density,'-',color='xkcd:purple',label='Planet model')
    if showPREM:
        axes[ai,aj].plot(PREM.radius/1.e3,PREM.density/1.e3,'-.',color='xkcd:deep blue',label='PREM (Earth)')
    axes[ai,aj].set_title(str(planet.M/Mearth)[0:6]+" (Mearth) Total Mass")

    #------------------------------
    ai=0
    aj=1
    axes[ai,aj].set_xlabel('Radius (km)')
    axes[ai,aj].set_ylabel('Pressure (GPa)')

    if inclpart:
        axes[ai,aj].scatter(particleplanet.r/1e5,particleplanet.P/1e10,s=6,c=particleplanet.S/1e7,vmax=4.)
    axes[ai,aj].plot(planet.rarr/1.e5,planet.pressure/1.e10,'-',color='xkcd:purple',label='Planet model')
    if showPREM:
        axes[ai,aj].plot(PREM.radius/1.e3,PREM.pressure,'-.',color='xkcd:deep blue',label='PREM (Earth)')
    axes[ai,aj].set_title(str(planet.cf)[0:6]+" Core Mass Fraction")

    #------------------------------
    ai=1
    aj=0
    axes[ai,aj].set_xlabel('Radius (km)')
    axes[ai,aj].set_ylabel('Temperature (K)')

    if inclpart:
        axes[ai,aj].scatter(particleplanet.r/1e5,particleplanet.T,s=6,c=particleplanet.S/1e7,vmax=4.)
    axes[ai,aj].plot(planet.rarr/1.e5,planet.temperature,'-',color='xkcd:purple',label='Planet model')
    if showPREM:
        axes[ai,aj].plot(PREM.radius/1.e3,PREM.temperature,'-.',color='xkcd:deep blue',label='Anzellini et al. 2013 (Earth)')

    #------------------------------
    ai=1
    aj=1

    axes[ai,aj].set_xlabel('Pressure (GPa)')
    axes[ai,aj].set_ylabel('Temperature (K)')
    if mEOS.mc.NT > 0:
        axes[ai,aj].plot(mEOS.mc.Pl,mEOS.mc.T,'-',color='black',label='Forsterite melt curve',markersize=10)
    if cEOS.mc.NT > 0:
        axes[ai,aj].plot(cEOS.mc.Pl,cEOS.mc.T,'--',color='black',label=coreEOS+' melt curve')

    if inclpart:
        axes[ai,aj].scatter(particleplanet.P/1e10,particleplanet.T,s=6,c=particleplanet.S/1e7,vmax=4.)
    axes[ai,aj].plot(planet.pressure/1.e10,planet.temperature,'-',color='xkcd:purple',label='Planet model')
    if showPREM:
        axes[ai,aj].plot(PREM.pressure,PREM.temperature,'-.',color='xkcd:deep blue',label='Anzellini et al. 2013 (Earth)')


    axes[ai,aj].set_ylim(0,1.2*max(planet.temperature))
    axes[ai,aj].set_xlim(0,1.2*max(planet.pressure/1.e10))

    #------------------------------
    ai=2
    aj=0

    axes[ai,aj].set_ylabel('Pressure (GPa)')
    axes[ai,aj].set_xlabel('Entropy (kJ K$^{-1}$ kg$^{-1}$)')

    if mEOS.mc.NT > 0:
        axes[ai,aj].plot(mEOS.mc.Sl*1e3,mEOS.mc.Pl,'-',color='black',label=mantleEOS+' PB')
        axes[ai,aj].plot(mEOS.mc.Ss*1e3,mEOS.mc.Ps,'-',color='black',markersize=10)
    if cEOS.mc.NT > 0:
        axes[ai,aj].plot(cEOS.mc.Sl*1e3,cEOS.mc.Pl,'--',color='black',label=coreEOS+' PB')
        axes[ai,aj].plot(cEOS.mc.Ss*1e3,cEOS.mc.Ps,'--',color='black')

    axes[ai,aj].plot(mEOS.vc.Sl*1e3,mEOS.vc.Pl,'-',color='black',markersize=10)
    axes[ai,aj].plot(cEOS.vc.Sl*1e3,cEOS.vc.Pl,'--',color='black',markersize=10)
    axes[ai,aj].plot(mEOS.vc.Sv*1e3,mEOS.vc.Pv,'-',color='black',markersize=10) #label='forsterite VD'
    axes[ai,aj].plot(cEOS.vc.Sv*1e3,cEOS.vc.Pv,'--',color='black',markersize=10) #label='iron VD'

    if inclpart:
        axes[ai,aj].scatter(particleplanet.S/1e7,particleplanet.P/1e10,s=6,c=particleplanet.S/1e7,vmax=4.)
    axes[ai,aj].plot(planet.entropy/1e7,planet.pressure/1.e10,'-',color='xkcd:purple',label='Planet model')

    axes[ai,aj].set_xlim(1,10.)
    axes[ai,aj].set_ylim(0,max(planet.pressure/1.e10))

    #------------------------------
    ai=2
    aj=1

    axes[ai,aj].set_ylabel('Pressure (GPa)')
    axes[ai,aj].set_xlabel('Entropy (kJ K$^{-1}$ kg$^{-1}$)')

    if mEOS.mc.NT > 0:
        axes[ai,aj].plot(mEOS.mc.Sl*1e3,mEOS.mc.Pl,'-',color='black',label=mantleEOS + ' PB')
        axes[ai,aj].plot(mEOS.mc.Ss*1e3,mEOS.mc.Ps,'-',color='black',markersize=10)
        axes[ai,aj].plot(mEOS.vc.Sl*1e3,mEOS.vc.Pl,'-',color='black',markersize=10)
        axes[ai,aj].plot(mEOS.vc.Sv*1e3,mEOS.vc.Pv,'-',color='black',markersize=10) #label='forsterite VD'
    if cEOS.mc.NT > 0:
        axes[ai,aj].plot(cEOS.mc.Sl*1e3,cEOS.mc.Pl,'--',color='black',label=coreEOS+' PB')
        axes[ai,aj].plot(cEOS.mc.Ss*1e3,cEOS.mc.Ps,'--',color='black')

        axes[ai,aj].plot(cEOS.vc.Sl*1e3,cEOS.vc.Pl,'--',color='black',markersize=10)
        axes[ai,aj].plot(cEOS.vc.Sv*1e3,cEOS.vc.Pv,'--',color='black',markersize=10) #label='iron VD'

    if inclpart:
        axes[ai,aj].scatter(particleplanet.S/1e7,particleplanet.P/1e10,s=6,c=particleplanet.S/1e7,vmax=4.)
    axes[ai,aj].plot(planet.entropy/1e7,planet.pressure/1.e10,'-',color='xkcd:purple',label='Planet model')

    axes[ai,aj].set_xlim(1,10.)
    axes[ai,aj].set_ylim(1e-5,max(planet.pressure/1.e10))
    axes[ai,aj].semilogy()

    # don't show a plot in lower right
    #axes[2,1].axis("off")

    axes[0,0].legend()
    axes[0,1].legend()
    axes[1,0].legend()
    axes[1,1].legend()
    axes[2,0].legend()
    axes[2,1].legend()

    # this saves a pdf file -- vector graphics are preferred
    if path:
        plt.savefig(path, format='pdf', dpi=300,transparent=True)

    if show:
        plt.show()
        


### in progress...
class EquilRun:
    def __init__(self):
        self.M = 0
        self.N = 0
        self.nsnaps = 0
        self.data = None
        self.prof1D = planet_profile()
        self.radp = npy.empty(0)
        self.radp1 = npy.empty(0)
        self.radmax = npy.empty(0)
        self.times = npy.empty(0)
        self.Sc = npy.empty(0)
        self.Sm = npy.empty(0)
        #P = npy.empty(0)
        #T = npy.empty(0)
        self.rhoc = npy.empty(0)
        self.rhom = npy.empty(0)
        self.ke = npy.empty(0)
        #ie = npy.empty(0)
        self.pe = npy.empty(0)
        self.v = npy.empty(0)
    
    def load(self, loc, thermo=False, inter=1, fname1D='planetprofile.dat',names1='equil_',names2='equilB_', icfile='planetIC.sw.hdf5'):
        """for gadget standard set names1 to 'snapshot_' and names2 to None"""
        N1 = N2 = 0
        f1 = f2 = []
        
        if names1:
            f1 = sorted(glob.glob(loc+names1+'*[!yml]'))
            N = [(f1[x].split('/')[-1]).split('_')[1].split('.')[0] for x in range(len(f1))]
            print(N)
            N1 = int(sorted(npy.array(N).astype(int))[-1])
        if names2:
            f2 = sorted(glob.glob(loc+names2+'*[!yml]'))
            N = [(f2[x].split('/')[-1]).split('_')[1].split('.')[0] for x in range(len(f2))]
            N2 = int(sorted(npy.array(N).astype(int))[-1])
        
        print(N1,N2)
        
        fnums = npy.arange(0,N1+N2,1)
        print(loc,fnums[::5])
        
        files = npy.array(f1+f2)[::inter]
        files = npy.append(npy.array(loc+icfile),files)
        self.nsnaps = len(files)
        self.data = npy.ndarray((self.nsnaps,),dtype=object)
        
        self.prof1D.load(loc+fname1D) 
        
        radp = []
        radp1 = []
        radmax = []
        times = []
        Sc = []
        Sm = []
        P = []
        T = []
        rhoc = []
        rhom = []
        ke = []
        pe = []
        ie = []
        v = []
        
        for i in range(self.nsnaps):
            honly = True #False #True
            if i==0 or i==1 or i==self.nsnaps:
                honly = False
            self.data[i] = Snapshot()
            print(files[i])
            self.data[i].load(files[i],headonly=honly,thermo=thermo)
            
            #rr = npy.sqrt( (s.x - (s.x[s.pot==s.pot.min()])[0])**2 + (s.y - (s.y[s.pot==s.pot.min()])[0])**2 + (s.z - (s.z[s.pot==s.pot.min()])[0])**2)
            #rr = npy.sqrt( (s.x-s.x.mean())**2 + (s.y-s.y.mean())**2 + (s.z-s.z.mean())**2 )
            rr = npy.sqrt( (self.data[i].x-npy.average(self.data[i].x,weights=self.data[i].m))**2 + (self.data[i].y-npy.average(self.data[i].y,weights=self.data[i].m))**2 + (self.data[i].z-npy.average(self.data[i].z,weights=self.data[i].m))**2 )
            radp.append(npy.median(npy.sort(rr)[-int(0.02*self.data[i].N):-1]))
            radp1.append(npy.median(npy.sort(rr)[-int(0.01*self.data[i].N):-1]))
            radmax.append(rr.max())
            #if npy.ndim(self.data[i].header.time)==0:
            #    times.append(self.data[i].header.time/3600.)
            #else:
            #    times.append(self.data[i].header.time[0]/3600.)
            Sc.append((self.data[i].S[self.data[i].id<GADGET_EOS_OFFSET]).mean())
            Sm.append((self.data[i].S[self.data[i].id>=GADGET_EOS_OFFSET]).mean())
            #P.append(s.P.mean())
            #T.append(s.T.mean())
            rhoc.append(self.data[i].rho[self.data[i].id<GADGET_EOS_OFFSET].mean())
            rhom.append(self.data[i].rho[self.data[i].id>=GADGET_EOS_OFFSET].mean())
            ke.append( (self.data[i].m*0.5*(self.data[i].vx**2 + self.data[i].vy**2 + self.data[i].vz**2)).sum() )
            #ie.append( (self.data[i].m*self.data[i].U).sum() )
            pe.append( (self.data[i].m * 0.5*self.data[i].pot).sum() )
            v.append( npy.sqrt( (self.data[i].vx**2 + self.data[i].vy**2 + self.data[i].vz**2).mean() ) ) #+s.vy**2+s.vz**2
        
        #print(times)
        self.radp = npy.array(radp)
        self.radp1 = npy.array(radp1)
        self.radmax = npy.array(radmax)
        self.times = npy.array(times)
        self.Sc = npy.array(Sc)
        self.Sm = npy.array(Sm)
        #P = npy.array(P)
        #T = npy.array(T)
        self.rhoc = npy.array(rhoc)
        self.rhom = npy.array(rhom)
        self.ke = npy.array(ke)
        #ie = npy.array(ie)
        self.pe = npy.array(pe)
        self.v = npy.array(v)
        
        self.vesc = npy.sqrt( 2*G*self.data[i].m.sum()/radmax[0] )
        #minpe = self.pe.min()
        #pe = pe - minpe
        
        #energy = ke+ie+pe
        

    
    def equil_mov_plot(n,scale='Mm'):
        recenter = True
        thermo=False
        ncol=4
        tmax = 24 #24. #12
        if times[-1]<tmax:
            tmax=times[-1]*1.01
        
        rhomin=5e-2 #5e5
        rhomax=10.
        Smin = 1.5
        Smax = 4. #10.
        
        zcutl = -0.2 #2.5
        zcuth = 0.2 #2.5
        vcut = 1e18
        
        lpos=16.6
        
        Ng=401j
        Ngz=61j
        #xmin=-8
        #xmax=8
        #ymin=-8
        #ymax=8
        #zmin=-8
        #zmax=8
        cmap = plt.get_cmap('plasma').copy()
        cmap.set_under('k')
        
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

        zmin=-axlim/10.
        zmax=axlim/10.
        zi=npy.linspace(zmin,zmax,int(Ngz.imag))
        X,Y,Z = npy.mgrid[-axlim:axlim:(Ng),-axlim:axlim:(Ng),zmin:zmax:(Ngz)]
        
        
        print(n)
        #ke[n] i = ( (self.data[i].m*0.5*(self.data[i].vx**2 + s.vy**2 + s.vz**2)).sum() )
        ##iei = ( (s.m*s.U).sum() )
        #pei = ( (s.m * 0.5*s.pot).sum() ) - minpe
        #if code=='gadget':
        #    ti = ( s.header.time )/3600.
        #elif code=='swift':
        #    ti = ( s.header.time )[0]/3600.
        #rr = npy.sqrt( (s.x-s.x.mean())**2 + (s.y-s.y.mean())**2 + (s.z-s.z.mean())**2 )
        #rr = npy.sqrt( (s.x-npy.average(s.x,weights=s.m))**2 + (s.y-npy.average(s.y,weights=s.m))**2 + (s.z-npy.average(s.z,weights=s.m))**2 )
        #rr = npy.sqrt( (s.x - (s.x[s.pot==s.pot.min()])[0])**2 + (s.y - (s.y[s.pot==s.pot.min()])[0])**2 + (s.z - (s.z[s.pot==s.pot.min()])[0])**2)
        #radpi = (npy.median(npy.sort(rr)[-int(0.02*s.N):-1]))
        #radp1i = (npy.median(npy.sort(rr)[-int(0.01*s.N):-1]))
        #radmaxi = (rr.max())
        #Sci = ((s.S[s.id<GADGET_EOS_OFFSET]).mean())
        #Smi = ((s.S[s.id>=GADGET_EOS_OFFSET]).mean())
        #rhoci = (s.rho[s.id<GADGET_EOS_OFFSET].mean())
        #rhomi = (s.rho[s.id>=GADGET_EOS_OFFSET].mean())
        #veli = ( npy.sqrt( (s.vx**2 + s.vy**2 + s.vz**2).mean() ) )
        #print(ti)
        
    
        #if recenter:
        x = self.data[i].x - (self.data[i].x[self.data[i].pot==self.data[i].pot.min()])[0]
        y = self.data[i].y - (self.data[i].y[self.data[i].pot==self.data[i].pot.min()])[0]
        z = self.data[i].z - (self.data[i].z[self.data[i].pot==self.data[i].pot.min()])[0]
        rr = npy.sqrt(x**2+y**2+z**2)
        
        vi = interpolate.griddata((x[self.data[i].vx<vcut]/scf,y[self.data[i].vx<vcut]/scf,z[self.data[i].vx<vcut]/scf),self.data[i].rho[self.data[i].vx<vcut],(X,Y,Z),method='linear',fill_value=1.e-18)
        
        coz = (z[self.data[i].pot==self.data[i].pot.min()])[0]
        nn=(npy.nonzero(zi==(zi[zi<=coz])[-1])[0])[0]
    
        
        plt.clf()
        
        # density plot (fluid)
        gs = gridspec.GridSpec(2, ncol)
        plt.subplot(gs[0, 0])
        #plt.subplot2grid((2, ncol), (0, 0))
        plt.minorticks_on()    
        
        im = plt.imshow(vi[:,:,nn].T,origin='lower',extent=[-axlim,axlim,-axlim,axlim],cmap=cmap,norm=matplotlib.colors.LogNorm(vmin=rhomin,vmax=rhomax,clip=False))
        plt.ylabel('$y$ (Mm)')
        ax=plt.gca()
        ax.tick_params(colors='w',which='both',labelcolor='k')
        ax.spines['top'].set_color('w')
        ax.spines['bottom'].set_color('w')
        ax.spines['left'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10.00))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.00))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10.00))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(2.00))
        ax.xaxis.set_ticklabels('')
        
        cbar_ax = fig.add_axes([0.240, 0.58, 0.008, 0.41])
        cbar = fig.colorbar(im, cax=cbar_ax,label=r'$\rho$ (g$\,$cm$^{-3}$)')
        cbar_ax.yaxis.set_label_coords(5.,0.5)
        #plt.colorbar(fraction=0.0466,pad=0.01,label=r'$\rho$ (g$\,$cm$^{-3}$)')
        
        # entropy plot (particle)
        plt.subplot(gs[1,0])
        #plt.subplot2grid((2, ncol), (1, 0))
        plt.minorticks_on()
        
        Splt = plt.scatter( (x)[(z>-zcut)*(z<zcut)]/scf, (y)[(z>-zcut)*(z<zcut)]/scf,s=0.5,marker='o',c=self.data[i].S[(z>-zcut)*(s.z<zcut)]/1e7,edgecolor='none',vmin=Smin,vmax=Smax)
        #plt.scatter( (s.x)[(s.z>zcutl)*(s.z<zcuth)*(s.id<GADGET_EOS_OFFSET)], (s.y)[(s.z>zcutl)*(s.z<zcuth)*(s.id<GADGET_EOS_OFFSET)],s=0.5,marker='o',c=s.S[(s.z>zcutl)*(s.z<zcuth)*(s.id<GADGET_EOS_OFFSET)]/1e7,edgecolor='none')
        plt.axis([-axlim,axlim,-axlim,axlim])
        plt.text(0.94,0.94,'{:.1f}'.format(ti),ha='right',va='top',fontsize=8,color='k',transform=plt.gca().transAxes)
        plt.ylabel('$y$ (Mm)')
        plt.xlabel('$x$ (Mm)')
        ax=plt.gca()
        ax.set_aspect(aspect=1.)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10.00))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.00))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10.00))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(2.00))
        cbar_ax = fig.add_axes([0.240, 0.1, 0.008, 0.41])
        cbar = fig.colorbar(Splt, cax=cbar_ax,label=r'$S$ (J$\,$K$^{-1}\,$g$^{-1}$)')
        cbar_ax.yaxis.set_label_coords(5.,0.5)
        
        gs.update(top=0.99,bottom=0.1,left=0.065,right=0.98,hspace=0.16,wspace=0.4)
        
        
        #radius plot
        gs2 = gridspec.GridSpec(2, ncol)
        plt.subplot(gs2[0, 1])
        #plt.subplot2grid((2, ncol), (0, 1))#, rowspan=2,colspan=2)
        plt.minorticks_on()
        
        plt.axvline(ti,c='0.5')
        plt.plot(self.data[:].header.time,(radp/(radp1[0]))[times<tmax],color='xkcd:deep rose',lw=1.,label='outer 2%') #,marker='+'
        plt.plot(times[times<tmax],(radp1/(radp1[0]))[times<tmax],color='xkcd:purple',lw=1.,label='outer 1%')
        plt.plot(times[times<tmax],(radmax/(radp1[0]))[times<tmax],color='xkcd:tangerine',lw=1.,label='max')
        
        plt.plot(ti,(radp[n]/(radp1[0])),ms=2,marker='o', color='xkcd:deep rose')
        plt.plot(ti,(radp1[n]/(radp1[0])),ms=2,marker='o', color='xkcd:purple')
        plt.plot(ti,(radmax[n]/(radp1[0])),ms=2,marker='o', color='xkcd:tangerine')
    
        #plt.ylim([0.,1.02])
        plt.xlim(xmin=0)
        plt.xlim(xmax=tmax)
        #plt.xlabel('Time (hrs)')
        plt.ylabel(u'R/R$_\mathrm{init}$')
        plt.legend()
        ax=plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5.))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.))
    
        
        #density profile plot
        plt.subplot(gs2[0, 2])
        plt.minorticks_on()
        
        plt.plot(planet1D.rarr/Rearth,planet1D.density,color='xkcd:purple',lw=1.) #,marker='+'
        plt.scatter(rr/Rearth,self.data[i].rho,s=1,marker='o', c=s.id, vmax=400000000)
        
        plt.ylim([0.,1.1*planet1D.density.max()])
        plt.xlim([0.,1.05*planet1D.rarr.max()/Rearth])
        #plt.xlabel('Radius (R$_\oplus$)')
        plt.ylabel(u'Density (g$\,$cm$^{-3}$)')
        
    
        #entropy profile plot
        plt.subplot(gs2[1, 2])
        plt.minorticks_on()
        
        plt.plot(planet1D.rarr/Rearth,planet1D.entropy/1e7,color='xkcd:purple',lw=1.) #,marker='+'
        plt.scatter(rr/Rearth,self.data[i].S/1e7,s=1,marker='o', c=s.id, vmax=400000000)
        
        plt.ylim([0.,1.2*planet1D.entropy.max()/1e7])
        plt.xlim([0.,1.05*planet1D.rarr.max()/Rearth])
        plt.xlabel('Radius (R$_\oplus$)')
        plt.ylabel(u'Entropy (J$\,$K$^{-1}\,$g$^{-1}$)')
        
    
        #velocity plot
        plt.subplot(gs2[0, 3])
        #plt.subplot2grid((2, ncol), (0, 2))#, rowspan=2,colspan=2)
        plt.minorticks_on()
        
        plt.axvline(ti,c='0.5')
        plt.plot(times[times<tmax],(v/vesc)[times<tmax],color='xkcd:purple',lw=1.) #,marker='+'
        
        plt.plot(ti,(veli/vesc),ms=2,marker='o', color='xkcd:purple')
        
        #plt.ylim([0.,1.02])
        plt.xlim(xmin=0)
        plt.xlim(xmax=tmax)
        plt.xlabel('Time (hrs)')
        plt.ylabel(u'v$_\mathrm{rms}$/v$_\mathrm{esc}$')
        plt.legend()
        ax=plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5.))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.))
    
    
        #mean entropy plot
        plt.subplot(gs2[1, 1])
        #plt.subplot2grid((2, ncol), (1, 1))#, rowspan=2,colspan=2)
        plt.minorticks_on()
        
        plt.axvline(ti,c='0.5')
        plt.plot(times[times<tmax],(Sm/Sm[0])[times<tmax],color='xkcd:purple',lw=1.,label='mantle') #,marker='+'
        plt.plot(times[times<tmax],(Sc/Sc[0])[times<tmax],color='xkcd:tangerine',lw=1.,label='core') #,marker='+'
        
        plt.plot(ti,(Smi/Sm[0]),ms=2,marker='o', color='xkcd:purple')
        plt.plot(ti,(Sci/Sc[0]),ms=2,marker='o', color='xkcd:tangerine')
        
        #plt.ylim([0.,1.02])
        plt.xlim(xmin=0)
        plt.xlim(xmax=tmax)
        plt.xlabel('Time (hrs)')
        plt.ylabel(u'S/S$_\mathrm{init}$')
        plt.legend()
        ax=plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5.))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.))
    
        
        #energy plot
        if thermo:
            plt.subplot(gs2[1, 3])
            #plt.subplot2grid((2, ncol), (1, 2))#, rowspan=2,colspan=2)
            plt.minorticks_on()
    
            plt.plot(times[times<23.97],(energy/(energy[0]))[times<23.97],color='k',lw=1.) #,marker='+'
            plt.plot(times[times<23.97],(pe/(energy[0]))[times<23.97],color='xkcd:purple',lw=1.)
            plt.plot(times[times<23.97],((ke+pe)/(energy[0]))[times<23.97],color='xkcd:tangerine',lw=1.)
    
            plt.fill_between(times[times<=ti],0./(energy[0]),(pe[times<=ti])/(energy[0]), color=cmap(0.15),alpha=0.65)
            plt.fill_between(times[times<=ti],pe[times<=ti]/(energy[0]),(ke+pe)[times<=ti]/(energy[0]), color=cmap(0.73),alpha=0.94)
            plt.fill_between(times[times<=ti],(ke+pe)[times<=ti]/(energy[0]),energy[times<=ti]/(energy[0]), color=cmap(0.43),alpha=0.82)
    
            plt.plot(ti,pei/(energy[0]),ms=2,marker='o', color='xkcd:purple')
            plt.plot(ti,(pei+kei)/(energy[0]),ms=2,marker='o', color='xkcd:tangerine')
            plt.plot(ti,(pei+kei+iei)/(energy[0]),ms=2,marker='o', color='xkcd:deep rose')
    
            #plt.text(lpos,(0.5*penergy[times>lpos])[0]/(energy[0]),'Potential',color='purple',size=8,ha='right',va='center')
            #plt.text(lpos,((0.5*kenergy+penergy)[times>lpos])[0]/(energy[0]),'Kinetic',color='orangered',size=8,ha='right',va='center')
            #plt.text(lpos,((0.5*ienergy+kenergy+penergy)[times>lpos])[0]/(energy[0]),'Internal',color='mediumvioletred',size=8,ha='right',va='center')
    
            #plt.ylim([0.,1.02])
            plt.xlim(xmin=0)
            plt.xlim(xmax=tmax)
            plt.xlabel('Time (hrs)')
            plt.ylabel('Fraction of event energy')
            ax=plt.gca()
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5.)) #2.5
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.))
    
        gs2.update(top=0.99,bottom=0.1,right=0.98,hspace=0.14,wspace=0.38)
