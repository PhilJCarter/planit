import pytest
import numpy as npy
from planit.globaldefs import *
from planit import Impact


def test_impactclass_creation():
    c = Impact()
    assert c.nsnaps ==  0


def test_impact_load_seq(tmp_reference_impact_seq):
    i = Impact()
    i.load(tmp_reference_impact_seq, prefix='snap', ndigits=3)
    
    assert i.nsnaps == len(i.snap) == 12
    assert i.data[0].N == i.data[6].N == 99514
    npy.testing.assert_array_equal(i.snap[1].m, i.data[9].m)


def test_impact_load_list(tmp_reference_impact_seq):
    i = Impact()
    i.load(tmp_reference_impact_seq, prefix='snap', ndigits=3, flist=[0, 2, 8], compress=False)

    assert i.nsnaps == len(i.snap) == 3
    assert i.data[0].N == i.data[2].N == 99514
    npy.testing.assert_array_equal(i.snap[1].S, i.data[2].S)


def test_impact_load_2seq(tmp_reference_impact_seq):
    i = Impact()
    i.load(tmp_reference_impact_seq, prefix='snap', prefix2='snap', ndigits=3)
    
    assert i.nsnaps == len(i.snap) == 24
    assert i.data[0].N == i.data[14].N == 99514
    npy.testing.assert_array_equal(i.snap[3].rho, i.data[20].rho)


