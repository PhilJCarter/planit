from ..main import *
from scipy import interpolate
import numpy as npy


def from_rhoT(Qlab,rho,T,EOS,dolog=True):

    if rho > 2.0:
        dolog = False

    if rho <= EOS.rho[0]:
        ir0 = 0
    else:
        ir0 = npy.where(EOS.rho<rho)[0][-1]
    if ir0 == len(EOS.rho)-1:
        ir0 -=1
    if T <= EOS.T[0]:
        iT0 = 0
    else:
        iT0 = npy.where(EOS.T<T)[0][-1]
    
    r0 = EOS.rho[ir0]
    r1 = EOS.rho[ir0+1]
    T0 = EOS.T[iT0]
    T1 = EOS.T[iT0+1]

    if Qlab == 'S':
        Qarr = EOS.S
    elif Qlab == 'P':
        Qarr = EOS.P
    elif Qlab == 'U':
        Qarr = EOS.U
    elif Qlab == 'A':
        Qarr = EOS.A
    elif Qlab == 'cs':
        Qarr = EOS.cs
    elif Qlab == 'cv':
        Qarr = EOS.cv
    else:
        raise ValueError('Unknown thermodynamic property:',Qlab)
    
    Q00 = Qarr[iT0,ir0]
    Q01 = Qarr[iT0,ir0+1]
    Q10 = Qarr[iT0+1,ir0]
    Q11 = Qarr[iT0+1,ir0+1]

    if dolog and min(Q00,Q01,Q10,Q11,rho,T)>0:
        Q00 = npy.log10(Q00)
        Q01 = npy.log10(Q01)
        Q10 = npy.log10(Q10)
        Q11 = npy.log10(Q11)

        rho = npy.log10(rho)
        T = npy.log10(T)
        r0 = npy.log10(r0)
        r1 = npy.log10(r1)
        T0 = npy.log10(T0)
        T1 = npy.log10(T1)
    else:
        dolog = False
        
    dr = rho - r0
    dT = T - T0

    Qa  = Q00 + dr*(Q01-Q00)/(r1-r0)
    Qb  = Q10 + dr*(Q11-Q10)/(r1-r0)
    Q  = Qa + dT*(Qb-Qa)/(T1-T0)

    if dolog:
        Q = 10**Q

    return Q



def from_rhoU(Qlab,rho,U,EOS,dolog=True):

    if rho > 2.0:
        dolog = False

    if rho <= EOS.rho[0]:
        ir0 = 0
    else:
        ir0 = npy.where(EOS.rho<rho)[0][-1]
    if ir0 == len(EOS.rho)-1:
        ir0 -=1
    if U <= EOS.U[0,ir0]:
        iU0r0 = 0
    else:
        iU0r0 = npy.where(EOS.U[:,ir0]<U)[0][-1]
    if U <= EOS.U[0,ir0+1]:
        iU0r1 = 0
    else:
        iU0r1 = npy.where(EOS.U[:,ir0+1]<U)[0][-1]

    if iU0r0 == len(EOS.U[:,ir0])-1:
        iU0r0 -=1
    if iU0r1 == len(EOS.U[:,ir0+1])-1:
        iU0r1 -=1
    
    r0 = EOS.rho[ir0]
    r1 = EOS.rho[ir0+1]
    U0r0 = EOS.U[iU0r0,ir0]
    U1r0 = EOS.U[iU0r0+1,ir0]
    U0r1 = EOS.U[iU0r1,ir0+1]
    U1r1 = EOS.U[iU0r1+1,ir0+1]
    
    if U < min(U0r0,U0r1):
        U = min(U0r0,U0r1)

    if Qlab == 'S':
        Qarr = EOS.S
    elif Qlab == 'P':
        Qarr = EOS.P
    elif Qlab == 'T':
        Qarr = EOS.T
    elif Qlab == 'A':
        Qarr = EOS.A
    elif Qlab == 'cs':
        Qarr = EOS.cs
    elif Qlab == 'cv':
        Qarr = EOS.cv
    else:
        raise ValueError('Unknown thermodynamic property:',Qlab)

    if Qlab == 'T':
        Q00 = Qarr[iU0r0]
        Q01 = Qarr[iU0r1]
        Q10 = Qarr[iU0r0+1]
        Q11 = Qarr[iU0r1+1]
    else:
        Q00 = Qarr[iU0r0,ir0]
        Q01 = Qarr[iU0r1,ir0+1]
        Q10 = Qarr[iU0r0+1,ir0]
        Q11 = Qarr[iU0r1+1,ir0+1]

    if dolog and min(Q00,Q01,Q10,Q11,rho,U)>0:
        Q00 = npy.log10(Q00)
        Q01 = npy.log10(Q01)
        Q10 = npy.log10(Q10)
        Q11 = npy.log10(Q11)

        rho = npy.log10(rho)
        U = npy.log10(U)
        r0 = npy.log10(r0)
        r1 = npy.log10(r1)
        U0r0 = npy.log10(U0r0)
        U1r0 = npy.log10(U1r0)
        U0r1 = npy.log10(U0r1)
        U1r1 = npy.log10(U1r1)
    else:
        dolog = False
        
    dr = rho - r0
    
    if U1r0==U0r0:
        Qa = Q00
    else:
        Qa  = (U1r0-U)*Q00/(U1r0-U0r0) + (U-U0r0)*Q10/(U1r0-U0r0)
    if U1r1==U0r1:
        Qb = Q01
    else:
        Qb  = (U1r1-U)*Q01/(U1r1-U0r1) + (U-U0r1)*Q11/(U1r1-U0r1)
    Q  = Qa + dr*(Qb-Qa)/(r1-r0)
    

    
    if dolog:
        Q = 10**Q

    return Q



def from_rhoS(Qlab,rho,S,EOS,dolog=True):

    if rho > 2.0:
        dolog = False

    ir0 = npy.where(EOS.rho<rho)[0][-1]
    iS0r0 = npy.where(EOS.S[:,ir0]<S)[0][-1]
    iS0r1 = npy.where(EOS.S[:,ir0+1]<S)[0][-1]
    
    r0 = EOS.rho[ir0]
    r1 = EOS.rho[ir0+1]
    S0r0 = EOS.S[iS0r0,ir0]
    S1r0 = EOS.S[iS0r0+1,ir0]
    S0r1 = EOS.S[iS0r1,ir0+1]
    S1r1 = EOS.S[iS0r1+1,ir0+1]

    if Qlab == 'U':
        Qarr = EOS.U
    elif Qlab == 'P':
        Qarr = EOS.P
    elif Qlab == 'T':
        Qarr = EOS.T
    elif Qlab == 'A':
        Qarr = EOS.A
    elif Qlab == 'cs':
        Qarr = EOS.cs
    elif Qlab == 'cv':
        Qarr = EOS.cv
    else:
        raise ValueError('Unknown thermodynamic property:',Qlab)

    if Qlab == 'T':
        Q00 = Qarr[iS0r0]
        Q01 = Qarr[iS0r1]
        Q10 = Qarr[iS0r0+1]
        Q11 = Qarr[iS0r1+1]
    else:
        Q00 = Qarr[iS0r0,ir0]
        Q01 = Qarr[iS0r1,ir0+1]
        Q10 = Qarr[iS0r0+1,ir0]
        Q11 = Qarr[iS0r1+1,ir0+1]

    if dolog:
        Q00 = npy.log10(Q00)
        Q01 = npy.log10(Q01)
        Q10 = npy.log10(Q10)
        Q11 = npy.log10(Q11)

        rho = npy.log10(rho)
        U = npy.log10(S)
        r0 = npy.log10(r0)
        r1 = npy.log10(r1)
        U0r0 = npy.log10(S0r0)
        U1r0 = npy.log10(S1r0)
        U0r1 = npy.log10(S0r1)
        U1r1 = npy.log10(S1r1)
        
    dr = rho - r0
    
    Qa  = (S1r0-S)*Q00/(S1r0-S0r0) + (S-S0r0)*Q10/(S1r0-S0r0)
    Qb  = (S1r1-S)*Q01/(S1r1-S0r1) + (S-S0r1)*Q11/(S1r1-S0r1)
    Q  = Qa + dr*(Qb-Qa)/(r1-r0)
    
    if dolog:
        Q = 10**Q

    return Q



def from_rhoU1D(Qlab,rho,U,EOS,dolog=True):

    #if rho > 2.0:
    #    dolog = False

    if rho <= EOS.rho[0]:
        ir0 = 0
    else:
        ir0 = npy.where(EOS.rho<rho)[0][-1]
    if ir0 == len(EOS.rho)-1:
        ir0 -=1
    if U <= EOS.U_1D[0]:
        iU0 = 0
    else:
        iU0 = npy.where(EOS.U_1D<U)[0][-1]
    
    r0 = EOS.rho[ir0]
    r1 = EOS.rho[ir0+1]
    U0 = EOS.U_1D[iU0]
    U1 = EOS.U_1D[iU0+1]

    if Qlab == 'P':
        Qarr = EOS.P
    elif Qlab == 'T':
        Qarr = EOS.T
    else:
        raise ValueError('Unknown thermodynamic property:',Qlab)
    
    Q00 = Qarr[iU0,ir0]
    Q01 = Qarr[iU0,ir0+1]
    Q10 = Qarr[iU0+1,ir0]
    Q11 = Qarr[iU0+1,ir0+1]

    if dolog and min(Q00,Q01,Q10,Q11,rho,U)>0:
        Q00 = npy.log10(Q00)
        Q01 = npy.log10(Q01)
        Q10 = npy.log10(Q10)
        Q11 = npy.log10(Q11)

        rho = npy.log10(rho)
        U = npy.log10(U)
        r0 = npy.log10(r0)
        r1 = npy.log10(r1)
        U0 = npy.log10(U0)
        U1 = npy.log10(U1)
    else:
        dolog = False
        
    dr = rho - r0
    dU = U - U0

    Qa  = Q00 + dr*(Q01-Q00)/(r1-r0)
    Qb  = Q10 + dr*(Q11-Q10)/(r1-r0)
    Q  = Qa + dU*(Qb-Qa)/(U1-U0)

    if dolog:
        Q = 10**Q

    return Q



