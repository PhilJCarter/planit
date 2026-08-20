import pytest
from planit import utils
import numpy as npy


def test_hmax_calc():
    assert utils.calc_hmax(1e-5, 1e-6*5.972e27) == pytest.approx(0.98895, rel=1e-4)


def test_com_calc():
    m = npy.array([1., 1., 1., 1., 2.])
    x = npy.array([1, 0., 0., 0., -2.])
    y = npy.array([0., -1., -2., 0., 0.])
    z = npy.array([0., 0., 0., 0., 0.])
    assert utils.com(m,x,y,z) == pytest.approx((-0.5,-0.5,0), rel=1e-5)


def test_com_v_calc():
    m = npy.array([1., 1., 1., 1., 2.])
    x = npy.array([1, 0., 0., 0., -2.])
    y = npy.array([0., -1., -2., 0., 0.])
    z = npy.array([0., 0., 0., 0., 0.])
    assert utils.com_v(m,x,y,z) == pytest.approx((-0.5,-0.5,0), rel=1e-5)


def test_calc_pot_tree():
    m = npy.array([1., 1.])
    x = npy.array([1., 0.])
    y = z = npy.array([0., 0.])
    assert utils._calc_potential_tree(m, x, y, z) == pytest.approx(npy.array([0., 0.]), rel=1e-4)


def test_calc_pot():
    m = npy.array([1., 1.])
    x = npy.array([0., 1.])
    y = z = npy.array([0., 0.])
    assert utils.calc_potential(m, x, y, z) == pytest.approx(npy.array([-6.6743e-8, -6.6743e-8]), rel=1e-4)


def test_calc_pot_tree_overlap():
    m = npy.array([1., 1.])
    x = y = z = npy.array([0., 0.])
    assert utils._calc_potential_tree(m, x, y, z) == pytest.approx(npy.array([0., 0.]), rel=1e-4)


def test_calc_pot_direct_overlap():
    m = npy.array([1., 1.])
    x = y = z = npy.array([0., 0.])
    assert utils._calc_potential_direct(m, x, y, z) == pytest.approx(npy.array([0., 0.]), rel=1e-4)
