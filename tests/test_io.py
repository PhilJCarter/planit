import pytest
import numpy as npy
from planit.globaldefs import *
from planit import Snapshot


def test_snapshot_load(reference_snapshot):
    s = Snapshot()
    s.load(reference_snapshot)
    
    assert s.N == 99514
    assert s.header.time/3600. == pytest.approx(24,rel=1e-5)
    assert s.m.sum()/Mearth == pytest.approx(0.099514,rel=1e-5)


def test_snapshot_load_headonly(reference_snapshot):
    s = Snapshot()
    s.load(reference_snapshot,headonly=True)
    
    npy.testing.assert_array_equal(npy.array([99514,0,0,0,0,0,0]),s.header.npart)
    assert s.materialIDs == None


def test_snasphot_hdf5_write(reference_snapshot,tmp_path):
    s0 = Snapshot()
    s0.load(reference_snapshot)
    s0.write(tmp_path / "test.hdf5")
    s1 = Snapshot()
    s1.load(tmp_path / "test.hdf5")
    
    assert (s0.id == s1.id).all()
    npy.testing.assert_array_equal(s0.hsml,s1.hsml)


def test_snasphot_G2_write(reference_snapshot,tmp_path):
    s0 = Snapshot()
    s0.load(reference_snapshot)
    s0.write(tmp_path / "test")
    s1 = Snapshot()
    s1.load(tmp_path / "test")
    
    assert (s0.id == s1.id).all()
    npy.testing.assert_array_equal(s0.rho,s1.rho)
