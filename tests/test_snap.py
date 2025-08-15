import pytest
from planit import Snapshot


def test_snapshot_creation():
    s = Snapshot()
    assert s.header.time == s.N == 0
