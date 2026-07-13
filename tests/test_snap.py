import pytest
import numpy as npy
from planit import Snapshot


def test_snapshot_creation():
    s = Snapshot()
    assert s.header.time == s.N == 0

def test_snapshot_matID():
    s = Snapshot()
    s.id = npy.array([12,400000257,200045678])
    s.ensure_matIDs([401,402,400])
    npy.testing.assert_array_equal(npy.array([401,400,402]),s.materialIDs)
