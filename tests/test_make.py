import pytest
from planit.makeplanet import *
import matplotlib
matplotlib.use('Agg')


def test_profile_creation():
    p = planet_profile()
    assert p.M == p.cf == 0.


def test_adiabatic_layer(blank_planet_profile):
    blank_planet_profile.rarr = npy.array([6000e5])
    blank_planet_profile.density = npy.array([5])
    blank_planet_profile.pressure = npy.array([8e10])
    blank_planet_profile.temperature = npy.array([2000])
    blank_planet_profile.add_adiabatic_layer(mass=0.01*Mearth, EOS=eos.select('HM80HHe'), Pmin = 1e2)
    assert blank_planet_profile.M == pytest.approx(0.01*Mearth, rel=1e-3)


def test_adiabatic_layer_nonHM80(blank_planet_profile):
    blank_planet_profile.rarr = npy.array([0])
    blank_planet_profile.density = npy.array([12])
    blank_planet_profile.pressure = npy.array([350e10])
    blank_planet_profile.temperature = npy.array([6000])
    blank_planet_profile.add_adiabatic_layer(mass=0.5*Mearth, EOS=eos.select('iron'))
    assert blank_planet_profile.M == pytest.approx(0.5*Mearth, rel=1e-3)
    

def test_adiabatic_layer_no_EOS(blank_planet_profile):
    with pytest.raises(ValueError):
        blank_planet_profile.add_adiabatic_layer(mass=0.1*Mearth, EOS=None)


def test_adiabatic_layer_no_starting_point(blank_planet_profile):
    with pytest.raises(ValueError, match='Central values must be provided'):
        blank_planet_profile.add_adiabatic_layer(mass=0.1*Mearth, EOS=eos.select('Fo'))


def test_adiabatic_layer_array_length(blank_planet_profile):
    with pytest.raises(ValueError, match='must have same length'):
        blank_planet_profile.rarr = npy.array([6000e5])
        blank_planet_profile.density = npy.array([5])
        blank_planet_profile.pressure = npy.array([8e10])
        blank_planet_profile.temperature = npy.array([2000, 3000])
        blank_planet_profile.add_adiabatic_layer(mass=0.01*Mearth, EOS=eos.select('HM80HHe'), Pmin = 1e2)


def test_isentropic_layer(blank_planet_profile, iron_isentrope):
    blank_planet_profile.rarr = npy.array([0])
    blank_planet_profile.density = npy.array([12])
    blank_planet_profile.pressure = npy.array([350e10])
    blank_planet_profile.temperature = npy.array([6000])
    blank_planet_profile.add_isentropic_layer(mass=0.5*Mearth, isentrope=iron_isentrope)
    assert blank_planet_profile.M == pytest.approx(0.5*Mearth, rel=1e-3)


def test_isentropic_layer_no_isentrope(blank_planet_profile):
    with pytest.raises(ValueError, match='isentrope must be defined'):
        blank_planet_profile.add_isentropic_layer(mass=0.1*Mearth, isentrope=None)


def test_isentropic_layer_no_starting_point(blank_planet_profile, iron_isentrope):
    with pytest.raises(ValueError, match='Central values must be provided'):
        blank_planet_profile.add_isentropic_layer(mass=0.1*Mearth, isentrope=iron_isentrope)


def test_isentropic_layer_array_length(blank_planet_profile, iron_isentrope):
    with pytest.raises(ValueError, match='must have same length'):
        blank_planet_profile.rarr = npy.array([0])
        blank_planet_profile.density = npy.array([12, 10])
        blank_planet_profile.pressure = npy.array([350e10])
        blank_planet_profile.temperature = npy.array([6000])
        blank_planet_profile.add_isentropic_layer(mass=0.1*Mearth, isentrope=iron_isentrope)


@pytest.mark.parametrize( ('m', 'rho'), [(1.0, 5.12), (10, 2.885), (200, 0.3481)] )
def test_density_estimate(m,rho):
    assert planet_density(m) == pytest.approx(rho, rel=1e-3)


def test_earth_profile_mass():
    p,isen1,isen2 = make_1D_planet(mass=Mearth, corefraction=0.325, Pmin=1.e6, Score=1.81, Smantle=3.02,
        mtolerance=8e-4, layer1='iron', layer2='forsterite', layers=[], S=[], mantlepotT=False,
        plot=False, fixcoreT=False, rhocent=None, verbose=False)
    assert p.M == pytest.approx(Mearth,rel=1e-3)


def test_earth_profile_radius():
    p,isen1,isen2 = make_1D_planet(mass=Mearth, corefraction=0.325, Pmin=1.e6, Score=1.6, Smantle=3.02,
        mtolerance=1e-3, layer1='alloy', layer2='forsterite', layers=[], S=[], mantlepotT=False,
        plot=False, fixcoreT=False, rhocent=12, verbose=True)
    assert p.rarr[-1] == pytest.approx(Rearth,rel=5e-2)


def test_high_mass_planet_profile():
    p,isen1,isen2 = make_1D_planet(mass=5*Mearth, corefraction=0.325, Pmin=1.e6, Score=1.81, Smantle=3.02,
        mtolerance=1e-3, layer1='iron', layer2='forsterite', layers=[], S=[], mantlepotT=False,
        plot=True, fixcoreT=False, rhocent=None, verbose=False)
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
        mtolerance=5e-3, layers=[402,'forsterite','HM80HHe'], S=[1.8,2.7,'adiabatic'], fixcoreT=False)
    assert p0.M == pytest.approx(Mearth, rel=5e-3)
    assert p0.cf == pytest.approx(0.3, rel=1e-3)
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
        mtolerance=1e-3, layers=['iron','forsterite'], S=[1.81,3.02], fixcoreT=False)
    sn,part = make_SPH_planet(profile=p0, layers=['iron','forsterite'], S=[1.81,3.02], resolution=res)
    assert sn.m.sum() == pytest.approx(p0.M, rel=2e-2)
    assert len(sn.id) == pytest.approx(res, rel=1e-2)


def test_PREM_load():
    p = PREMclass()
    assert p.NR == 504
    assert p.radius[-1] == pytest.approx(6.371e6, rel=1e-5)
