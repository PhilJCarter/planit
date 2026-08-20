import pytest
import numpy as npy
import seagen
from planit.globaldefs import *
from planit import Snapshot


def test_snapshot_load(reference_snapshot):
    s = Snapshot()
    s.load(reference_snapshot)
    
    assert s.N == 99514
    assert s.header.time/3600. == pytest.approx(24,rel=1e-5)
    assert s.m.sum()/Mearth == pytest.approx(0.099514,rel=1e-5)


def test_snapshot_load_thermo(reference_snapshot):
    s = Snapshot()
    s.load(reference_snapshot, thermo=True)
    
    assert s.N == 99514
    assert len(s.T) == s.N
    assert s.m.sum()/Mearth == pytest.approx(0.099514,rel=1e-5)


def test_snapshot_load_compress(reference_snapshot):
    s = Snapshot()
    s.load(reference_snapshot, thermo=True, compress=True)
    
    assert s.N == 99514
    assert len(s.P) == s.N
    assert s.m.sum()/Mearth == pytest.approx(0.099514,rel=1e-5)


def test_snapshot_load_headonly(reference_snapshot):
    s = Snapshot()
    s.load(reference_snapshot,headonly=True)
    
    npy.testing.assert_array_equal(npy.array([99514,0,0,0,0,0,0]),s.header.npart)
    assert s.materialIDs == None


def test_snasphot_hdf5_write(reference_snapshot, tmp_path):
    s0 = Snapshot()
    s0.load(reference_snapshot)
    s0.write(tmp_path / "test.hdf5")
    s1 = Snapshot()
    s1.load(tmp_path / "test.hdf5")
    
    assert (s0.id == s1.id).all()
    npy.testing.assert_array_equal(s0.hsml,s1.hsml)


def test_snasphot_G2_write(reference_snapshot, tmp_path):
    s0 = Snapshot()
    s0.load(reference_snapshot)
    s0.write(tmp_path / "test")
    s1 = Snapshot()
    s1.load(tmp_path / "test")
    
    assert (s0.id == s1.id).all()
    npy.testing.assert_array_equal(s0.rho, s1.rho)


def test_snasphot_G2_write_thermo(reference_snapshot, tmp_path):
    s0 = Snapshot()
    s0.load(reference_snapshot, thermo=True)
    s0.write(tmp_path / "test")
    s1 = Snapshot()
    s1.load(tmp_path / "test", thermo=True)
    
    assert (s0.id == s1.id).all()
    npy.testing.assert_allclose(s0.U, s1.U, rtol=5e-5)


def test_load_seagen():
    radii = npy.arange(0.01, 10, 0.01)
    densities = npy.ones(len(radii)) 
    temperature = npy.ones(len(radii)) * 2000
    pressure = npy.ones(len(radii)) * 1e8
    materials = npy.zeros(len(radii))
    particles = seagen.GenSphere(10000, radii, densities, A1_T_prof=temperature, A1_P_prof=pressure , A1_mat_prof=materials)
    setattr(particles, "S", npy.ones(len(particles.mat))*3e7)
    s = Snapshot()
    s.ic_from_seagen(particles)
    
    assert s.N == len(particles.m)
    npy.testing.assert_array_equal(s.P, particles.P)