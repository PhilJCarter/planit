import pytest
from planit.makeplanet import *


def test_profile_creation():
    p = planet_profile()
    assert p.M == p.cf == 0.
    
def test_earth_mass():
    p,isen1,isen2 = make_1D_planet(mass=Mearth, corefraction=0.325, Pmin=1.e6, Score=1.81, Smantle=3.02,
        mtolerance=1e-3, layer1='iron', layer2='forsterite', layers=[], S=[], mantlepotT=False,
        plot=False, fixcoreT=False, rhocent=None, verbose=False)
    assert p.M == pytest.approx(Mearth,rel=1e-3)

def test_earth_radius():
    p,isen1,isen2 = make_1D_planet(mass=Mearth, corefraction=0.325, Pmin=1.e6, Score=1.81, Smantle=3.02,
        mtolerance=1e-3, layer1='alloy', layer2='forsterite', layers=[], S=[], mantlepotT=False,
        plot=False, fixcoreT=False, rhocent=None, verbose=False)
    assert p.rarr[-1] == pytest.approx(Rearth,rel=5e-2)

def test_mars_SPH_mass():
    p,c,m,sn,part = make_SPH_planet(mass=0.1*Mearth, corefraction=0.3, Pmin=1.e6, Score=1.81, Smantle=3.03, 
        mtolerance=1e-3, layer1='alloy', layer2='forsterite', layers=[], S=[], resolution=1e6)
    assert sn.m.sum() == pytest.approx(0.1*Mearth,rel=1e-2)
