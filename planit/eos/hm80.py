import scipy
import numpy as npy

HM80_c1 = 2.3638
HM80_c2 = -4.9842e-5
HM80_c3 = 1.1788e-8
HM80_c4 = -3.8101e-4
HM80_c5 = 2.6182
HM80_c6 = 0.45053
    
HM80_FMW = 2.2857143


def HM80_U_rho_T(rho, T):
    #return HM80_Cv_rho_T(rho,T)*T  # Keggerreis original
    # Lock & Stewart 2024:
    return 8.31446e7/HM80_FMW * ( HM80_c1*T + HM80_c2*T**2/2. + HM80_c3*T**3/3. + HM80_c4*rho*T**2/2. + HM80_c5*rho*T + HM80_c6*rho**2*T )


def HM80_Cv_rho_T(rho, T):
        return 8.31446e7/HM80_FMW * ( HM80_c1 + HM80_c2*T + HM80_c3*T**2 + HM80_c4*rho*T + HM80_c5*rho + HM80_c6*rho**2 )

def HM80_P_rho_T(rho, T):
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

    return npy.exp(lnP) * 1e6*1e6


def HM80_rho_P_T(P, T, rho_est=0.1, tolerance=1e-5):
    def objective(rho):
        return HM80_P_rho_T(rho, T) - P

    rho_sol = scipy.optimize.fsolve(objective, x0=rho_est, xtol=tolerance)
    return rho_sol[0]


def HM80_gamma_rho_T(rho, T):
    rho0 = 0.005
    y = npy.log(T)
    x = npy.log(rho/rho0)
    return HM80_gamma_lnrho_lnT(x,y)


def HM80_gamma_lnrho_lnT(x, y):
    b1 = 0.328471
    b2 = 0.0286529
    b3 = -0.00139609
    b4 = -0.0232258
    b5 = 0.0579055
    b6 = 0.0454488
    return b1 + b2*y + b3*y**2 + b4*x*y + b5*x + b6*x**2


def HM80_adiabat_T(rho0, rho1, T0):
    rhoref = 0.005
    sol = scipy.integrate.solve_ivp(HM80_gamma_lnrho_lnT,[npy.log(rho0/rhoref),npy.log(rho1/rhoref)],[npy.log(T0)])
    return npy.exp(sol.y[0][-1])


