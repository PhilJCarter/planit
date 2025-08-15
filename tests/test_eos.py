import pytest
import random
import numpy as npy
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

def test_isentrope_init():
    i = eos.isentrope_class()
    assert i.entropy is None

def test_calcprop_unknown():
    with pytest.raises(Exception):
        eos.calcprop('3','rho','T',4,3000,401)


@pytest.mark.parametrize('execcount', range(500))
def test_interp_ANEOS_U(execcount):
    aneoslist = ['ANEOSIron','ANEOSForsterite','ANEOSFeSiAlloy','5PhaseWater']
    EOS = eos.select(random.choice(aneoslist))
    j = npy.random.randint(0,high=len(EOS.rho))
    i = npy.random.randint(0,high=len(EOS.T))
    print(EOS.MODELNAME,j,i)
    assert eos.tabinterp.from_rhoT('U',EOS.rho[j]*(1.+1e-8),EOS.T[i]*(1.+1e-12),EOS) == pytest.approx(EOS.U[i,j], rel=1e-3, abs=1e-11)