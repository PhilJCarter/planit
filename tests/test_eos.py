import pytest
from planit import eos


def test_iron_eos_loading():
    assert eos.select('iron') is not None

def test_ironalloy_eos_loading():
    assert eos.select('FeSi') is not None

def test_forsterite_eos_loading():
    assert eos.select('ANEOSForsterite') is not None

def test_unknown_eos_loading():
    with pytest.raises(ValueError):
        eos.select('Cheese')


def test_calcprop_unknown():
    with pytest.raises(Exception):
        eos.calcprop('3','rho','T',4,3000,401)
