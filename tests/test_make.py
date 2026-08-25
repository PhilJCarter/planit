import pytest
from planit.makeplanet import *


def test_profile_creation():
    p = planet_profile()
    assert p.M == p.cf == 0.
    

def test_earth_profile_mass():
    p,isen1,isen2 = make_1D_planet(mass=Mearth, corefraction=0.325, Pmin=1.e6, Score=1.81, Smantle=3.02,
        mtolerance=1e-3, layer1='iron', layer2='forsterite', layers=[], S=[], mantlepotT=False,
        plot=False, fixcoreT=False, rhocent=None, verbose=False)
    assert p.M == pytest.approx(Mearth,rel=1e-3)


def test_earth_profile_radius():
    p,isen1,isen2 = make_1D_planet(mass=Mearth, corefraction=0.325, Pmin=1.e6, Score=1.81, Smantle=3.02,
        mtolerance=1e-3, layer1='alloy', layer2='forsterite', layers=[], S=[], mantlepotT=False,
        plot=False, fixcoreT=False, rhocent=None, verbose=False)
    assert p.rarr[-1] == pytest.approx(Rearth,rel=5e-2)


def test_high_mass_planet_profile():
    p,isen1,isen2 = make_1D_planet(mass=5*Mearth, corefraction=0.325, Pmin=1.e6, Score=1.81, Smantle=3.02,
        mtolerance=1e-3, layer1='iron', layer2='forsterite', layers=[], S=[], mantlepotT=False,
        plot=False, fixcoreT=False, rhocent=None, verbose=False)
    assert p.M == pytest.approx(5*Mearth,rel=1e-3)


def test_2_layer_planet_creation_alternatives():
    p0,isen1,isen2 = make_1D_planet(mass=2*Mearth, corefraction=0.3, Pmin=1.e5, Score=1.81, Smantle=3.02,
        mtolerance=1e-3, layer1='iron', layer2='forsterite', layers=[], S=[], fixcoreT=True)
    p1,isen3,isen4 = make_1D_planet(mass=[0.6*Mearth,1.4*Mearth], corefraction=0.3, Pmin=1.e5,
        mtolerance=1e-3, layers=['iron','forsterite'], S=[1.81,3.02], fixcoreT=True)
    assert p1.M == pytest.approx(p0.M, rel=1e-4)
    assert p1.cf == pytest.approx(p0.cf, rel=5e-4) == pytest.approx(0.3, rel=5e-4) 


def test_multi_layer_planet_creation():
    p0,isentropes = make_1D_planet(mass=[0.3*Mearth,0.65*Mearth,0.05*Mearth], Pmin=1.e6,
        mtolerance=1e-3, layers=[402,'forsterite','SS08'], S=[1.81,3.02,1.6], fixcoreT=False)
    assert p0.M == pytest.approx(Mearth, rel=2e-3)
    assert p0.cf == pytest.approx(0.3, rel=5e-4)
    assert isentropes[2].material == 'SS08'


def test_multi_layer_planet_wadiabatic_layer_creation():
    p0,isentropes = make_1D_planet(mass=[0.3*Mearth,0.64*Mearth,0.06*Mearth], Pmin=1.e6,
        mtolerance=1e-3, layers=[402,'forsterite','HM80HHe'], S=[1.8,2.7,'adiabatic'], fixcoreT=False)
    assert p0.M == pytest.approx(Mearth, rel=2e-3)
    assert p0.cf == pytest.approx(0.3, rel=5e-4)
    assert isentropes[2] == 'adiabat'


def test_multi_layer_planet_creation_list_error():
    with pytest.raises(ValueError):
        p1,isen1,isen2 = make_1D_planet(mass=2*Mearth, corefraction=0.3, Pmin=1.e5,
            mtolerance=1e-4, layers=['iron','forsterite'], S=[1.81,3.02,3.4], fixcoreT=True)


def test_mars_SPH_mass():
    p,c,m,sn,part = make_SPH_planet(mass=0.1*Mearth, corefraction=0.3, Pmin=1.e6, Score=1.81, Smantle=2.7, 
        mtolerance=1e-3, layer1='alloy', layer2='forsterite', layers=[], S=[], resolution=1e6)
    assert sn.m.sum() == pytest.approx(0.1*Mearth,rel=1e-2)


@pytest.mark.parametrize('res', [1e6, 1e7])
def test_SPH_from_1D_profile(res):
    p0,isen1,isen2 = make_1D_planet(mass=[0.3*Mearth,0.7*Mearth], corefraction=0.3, Pmin=1.e5,
        mtolerance=1e-4, layers=['iron','forsterite'], S=[1.81,3.02], fixcoreT=False)
    sn,part = make_SPH_planet(profile=p0, layers=['iron','forsterite'], S=[1.81,3.02], resolution=res)
    assert sn.m.sum() == pytest.approx(p0.M, rel=2e-2)
    assert len(sn.id) == pytest.approx(res, rel=1e-2)
