"""
   planit EOS table interpolation functions
"""

from ..main import *
from scipy import interpolate
import numpy as npy
import numba

@numba.njit
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
    if iT0 == len(EOS.T)-1:
        iT0 -=1
    
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



@numba.njit
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
        Tarr = EOS.T
    elif Qlab == 'A':
        Qarr = EOS.A
    elif Qlab == 'cs':
        Qarr = EOS.cs
    elif Qlab == 'cv':
        Qarr = EOS.cv
    else:
        raise ValueError('Unknown thermodynamic property:',Qlab)

    if Qlab == 'T':
        Q00 = Tarr[iU0r0]
        Q01 = Tarr[iU0r1]
        Q10 = Tarr[iU0r0+1]
        Q11 = Tarr[iU0r1+1]
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


@numba.njit
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
        Tarr = EOS.T
    elif Qlab == 'A':
        Qarr = EOS.A
    elif Qlab == 'cs':
        Qarr = EOS.cs
    elif Qlab == 'cv':
        Qarr = EOS.cv
    else:
        raise ValueError('Unknown thermodynamic property:',Qlab)

    if Qlab == 'T':
        Q00 = Tarr[iS0r0]
        Q01 = Tarr[iS0r1]
        Q10 = Tarr[iS0r0+1]
        Q11 = Tarr[iS0r1+1]
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


@numba.njit
def from_gadget_rhoS(Qlab, rho, S, EOS):
    """Interpolate one GADGET property at density ``rho`` and entropy ``S``.

    GADGET tables use entropy as an independent axis and store temperature as
    a two-dimensional property.  Arrays on ``EOS`` are indexed as
    ``[entropy_index, density_index]``.  Density is in g cm^-3; entropy and the
    requested property are already in PlanIt's internal table units because
    ``calcprop`` performs conversion at the public cgs interface.

    The optional logarithmic mode reproduces a convention used by some
    planetary GADGET builds at densities up to and including 2 g cm^-3.  It is
    not assumed to be a general property of every GADGET EOS.  Queries outside
    the table either raise an error or use the nearest endpoint, according to
    the policy stored in the EOS passer when its user slot was loaded.

    Parameters
    ----------
    Qlab : str
        Property to return: ``P``, ``T``, ``U``, or ``cs``.
    rho, S : float
        Query density and entropy in the internal units described above.
    EOS : EOSpasser
        Numba-compatible table created by ``GADtable.make_passer_class``.

    Returns
    -------
    float
        Interpolated property in PlanIt's internal table units.
    """
    # NaN and infinity cannot be converted into a meaningful endpoint value,
    # even in clip mode, so reject them before any comparisons or indexing.
    if not npy.isfinite(rho) or not npy.isfinite(S):
        raise ValueError('GADGET EOS density and entropy must be finite.')

    # Apply the load-time policy before clamping.  In strict mode, leaving the
    # simulated EOS domain is an analysis error.  Clip mode is an explicit
    # PlanIt nearest-boundary override and never extrapolates.
    outside_domain = (
        rho < EOS.rho[0]
        or rho > EOS.rho[EOS.ND - 1]
        or S < EOS.S[0, 0]
        or S > EOS.S[EOS.NT - 1, 0]
    )
    if EOS.gadget_out_of_domain == 'error':
        if outside_domain:
            raise ValueError('GADGET EOS query is outside the table domain.')
    elif EOS.gadget_out_of_domain != 'clip':
        raise ValueError(
            "GADGET EOS out-of-domain policy must be 'error' or 'clip'."
        )

    # Clamp after the strict check to implement the explicit clip policy.  In
    # strict mode these min/max operations are no-ops; the index bounds below
    # are what map exact first/last axis values onto adjacent valid 2x2 cells.
    rho = min(max(rho, EOS.rho[0]), EOS.rho[EOS.ND - 1])
    S = min(max(S, EOS.S[0, 0]), EOS.S[EOS.NT - 1, 0])

    # searchsorted(...)-1 gives the lower node of the bracketing density cell.
    # Move the two endpoints explicitly onto the first/last valid cell.
    ir0 = npy.searchsorted(EOS.rho, rho) - 1
    if ir0 < 0:
        ir0 = 0
    elif ir0 >= EOS.ND - 1:
        ir0 = EOS.ND - 2

    # GADtable's one-dimensional entropy axis is repeated across density when
    # constructing the generic EOSpasser, so every column is identical and
    # the first column recovers the direct axis.
    entropy_axis = EOS.S[:, 0]
    iS0 = npy.searchsorted(entropy_axis, S) - 1
    if iS0 < 0:
        iS0 = 0
    elif iS0 >= EOS.NT - 1:
        iS0 = EOS.NT - 2

    # Select the dependent-property grid.  Temperature is a two-dimensional
    # T(S, rho) output for this format and therefore uses T_2D; EOS.T is a
    # one-dimensional temperature axis for other PlanIt table types.
    if Qlab == 'P':
        Qarr = EOS.P
    elif Qlab == 'T':
        Qarr = EOS.T_2D
    elif Qlab == 'U':
        Qarr = EOS.U
    elif Qlab == 'cs':
        Qarr = EOS.cs
    else:
        raise ValueError('Unknown GADGET thermodynamic property.')

    # Read one interpolation cell.  The first property index is entropy and
    # the second is density:
    #
    #                 rho r0       rho r1
    # entropy S0        Q00          Q01
    # entropy S1        Q10          Q11
    r0 = EOS.rho[ir0]
    r1 = EOS.rho[ir0 + 1]
    S0 = entropy_axis[iS0]
    S1 = entropy_axis[iS0 + 1]
    Q00 = Qarr[iS0, ir0]
    Q01 = Qarr[iS0, ir0 + 1]
    Q10 = Qarr[iS0 + 1, ir0]
    Q11 = Qarr[iS0 + 1, ir0 + 1]

    # Some planetary GADGET builds switch from log10 interpolation to ordinary
    # linear interpolation above exactly 2 g cm^-3.  The boolean is explicit
    # because other GADGET simulations may use a linear table at all densities.
    use_log = EOS.gadget_low_density_log and rho <= 2.0
    if use_log:
        # rho is guaranteed positive by table validation and clipping.  Entropy
        # and all four property corners must also be positive before taking
        # logarithms; a negative tensile pressure cell, for example, remains
        # valid for linear interpolation but cannot be evaluated in log mode.
        if (S <= 0.0 or S0 <= 0.0 or S1 <= 0.0
                or Q00 <= 0.0 or Q01 <= 0.0
                or Q10 <= 0.0 or Q11 <= 0.0):
            raise ValueError(
                'Logarithmic GADGET interpolation requires positive entropy '
                'and property values throughout the selected table cell.'
            )
        # Form interpolation fractions in log10(rho) and log10(S), and
        # interpolate log10(Q), matching the selected GADGET convention.
        wr = ((npy.log10(rho) - npy.log10(r0))
              / (npy.log10(r1) - npy.log10(r0)))
        wS = ((npy.log10(S) - npy.log10(S0))
              / (npy.log10(S1) - npy.log10(S0)))
        Q00 = npy.log10(Q00)
        Q01 = npy.log10(Q01)
        Q10 = npy.log10(Q10)
        Q11 = npy.log10(Q11)
    else:
        # Ordinary bilinear weights in the native rho--S coordinates.
        wr = (rho - r0) / (r1 - r0)
        wS = (S - S0) / (S1 - S0)

    # First interpolate along density on the lower and upper entropy rows,
    # then interpolate those two values along entropy.  In log mode these are
    # logarithmic property values and the final exponentiation restores Q.
    Qa = Q00 + wr * (Q01 - Q00)
    Qb = Q10 + wr * (Q11 - Q10)
    Q = Qa + wS * (Qb - Qa)
    if use_log:
        Q = 10.0**Q
    return Q



@numba.njit
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
        Qarr = EOS.T_2D
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
