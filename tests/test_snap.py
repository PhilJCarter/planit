import pytest
import numpy as npy
from planit.globaldefs import *
from planit import Snapshot


def test_snapshot_creation():
    s = Snapshot()
    assert s.header.time == s.N == 0

def test_snapshot_matID():
    s = Snapshot()
    s.id = npy.array([12,400000257,200045678])
    s.ensure_matIDs([401,402,400])
    npy.testing.assert_array_equal(npy.array([401,400,402]),s.materialIDs)

def test_phase_calculation(snasphot_class_loaded):
    snasphot_class_loaded.calc_phase()
    assert (snasphot_class_loaded.meltfrac*snasphot_class_loaded.m).sum()/snasphot_class_loaded.m.sum() == pytest.approx(0.3,rel=1e-4)
    assert snasphot_class_loaded.phase[20341] == pytest.approx(6.0,rel=1e-10)


def test_equilibration_check(snasphot_class_loaded):
    assert snasphot_class_loaded.eq_test(threshold=0.001, verbose=True)


def test_particle_removal(snasphot_class_loaded):
    s = snasphot_class_loaded
    print(s.N,len(s.m))
    s.remove(npy.random.choice(s.id))
    print(s.N,len(s.m))
    assert s.N == len(s.m) == 99513


def test_particle_removal_G2(reference_snapshot,tmp_path):
    s0 = Snapshot()
    s0.load(reference_snapshot)
    s0.header.flag_entr_ics = 1
    s0.write(tmp_path / "test")
    s1 = Snapshot()
    s1.load(tmp_path / "test")
    print(s1.N,len(s1.m),len(s1.S))
    s1.remove(npy.random.choice(s1.id))
    print(s1.N,len(s1.m))
    assert s1.N == len(s1.m) == 99513


def test_particle_removal(reference_snapshot):
    with pytest.raises(ValueError):
        s = Snapshot()
        s.load(reference_snapshot)
        s.remove(-396)


def test_bound_mass_single_body(snasphot_class_loaded):
    s = snasphot_class_loaded
    s.bound_mass(nrem=5, save=False)
    assert s.m[s.rem==1].sum() == s.m.sum()
    assert len(s.id[s.rem==0]) == 0


def test_bound_mass_save(tmp_reference_snapshot):
    s0 = Snapshot()
    s0.load(tmp_reference_snapshot)
    s0.bound_mass(discardsmall=True, save=True)
    
    s1 = Snapshot()
    s1.load(tmp_reference_snapshot)
    
    npy.testing.assert_array_equal(s0.rem, s1.rem)


def test_combine(reference_snapshot):
    s1 = Snapshot()
    s1.load(reference_snapshot)
    s2 = Snapshot()
    s2.load(reference_snapshot)
    s1.combine(s1,s2)
    
    assert s1.N == 2*s2.N
    assert len(s1.vx) == 2*len(s2.vy)


def test_combine_incompatible(reference_snapshot):
    s1 = Snapshot()
    s1.load(reference_snapshot)
    s2 = Snapshot()
    s2.load(reference_snapshot)
    s2.header.flag_entr_ics = 1
    with pytest.raises(ValueError):
        s1.combine(s1,s2)


def test_correct_entropy_dict(reference_snapshot):
    s = Snapshot()
    s.load(reference_snapshot, thermo=True)
    S0 = s.S.copy()
    s.correct_entropy({402:1.81, 'forsterite':3.03})
    npy.testing.assert_allclose(s.S, S0, rtol=5e-2)


def test_correct_entropy_snap(reference_snapshot):
    s = Snapshot()
    s.load(reference_snapshot, thermo=True)
    S0 = s.S.copy()
    s.correct_entropy(s)
    npy.testing.assert_allclose(s.S, S0, rtol=1e-3)


def test_correct_entropy_type(reference_snapshot):
    s = Snapshot()
    s.load(reference_snapshot, thermo=True)
    with pytest.raises(TypeError):
        s.correct_entropy([1.81, 3.03])


def test_correct_entropy_dict_len(reference_snapshot):
    s = Snapshot()
    s.load(reference_snapshot, thermo=True)
    with pytest.raises(ValueError):
        s.correct_entropy({1:1.81, 2:3.03, 3:5.3})

