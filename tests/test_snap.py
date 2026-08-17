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

def test_phase_calculation(reference_snapshot):
    s = Snapshot()
    s.load(reference_snapshot,thermo=True)
    s.calc_phase()
    assert (s.meltfrac*s.m).sum()/s.m.sum() == pytest.approx(0.3,rel=1e-4)
    assert s.phase[20341] == pytest.approx(6.0,rel=1e-10)

def test_equilibration_check(reference_snapshot):
    s = Snapshot()
    s.load(reference_snapshot)
    assert s.eq_test(threshold=0.001)

