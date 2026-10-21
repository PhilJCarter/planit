import pytest
import numpy as npy
import matplotlib
matplotlib.use('Agg')
import IPython
from planit.globaldefs import *
from planit import Impact
from planit import plotting

@pytest.mark.parametrize(
    ('scale', 'fac'),
    [
        ('Mm', 1e8),
        ('km', 1e5),
        ('earth', Rearth),
        (12.35, 12.35),
    ],)
def test_get_scale_fac(scale,fac):
    assert plotting.get_scale_fac(scale) == fac


@pytest.mark.parametrize(
    ('scale', 'u_str'),
    [
        ('Mm', ' (Mm)'),
        ('km', ' (km)'),
        ('Earth', r' (R$_\oplus$)'),
        (3405.7, ''),
    ],)
def test_get_axis_units(scale,u_str):
    assert plotting.get_axis_units(scale) == u_str


def test_impact_plotseq_material(impact_class_loaded):
    fig = impact_class_loaded.plotseq(n=4,type='mats',zoom=1.2)
    assert isinstance(fig, matplotlib.pyplot.Figure)


def test_impact_plotseq_phase(impact_class_loaded):
    fig = impact_class_loaded.plotseq(n=3,ptype='phase')
    assert isinstance(fig, matplotlib.pyplot.Figure)


def test_impact_plotseq_entropy(impact_class_loaded):
    fig = impact_class_loaded.plotseq(n=2,type='S',focus='targcore',scale='Earth',zoom=1.0)
    assert isinstance(fig, matplotlib.pyplot.Figure)


def test_impact_plotseq_density(impact_class_loaded):
    fig = impact_class_loaded.plotseq(n=2,ptype='rho',focus='potmin',scale='km',zoom=1.5)
    assert isinstance(fig, matplotlib.pyplot.Figure)


def test_impact_mov_material_single(impact_class_loaded):
    fig = impact_class_loaded.attrmov(seq=[0,],ptype='mats',zoom=1.2)
    assert isinstance(fig, matplotlib.pyplot.Figure)


def test_impact_mov_pressure(impact_class_loaded):
    fig = impact_class_loaded.attrmov(seq=[0,2,6],ptype='P',zoom=1.8)
    assert isinstance(fig, IPython.display.Video)


def test_impact_mov_temperature(impact_class_loaded):
    fig = impact_class_loaded.attrmov(n=3,ptype='temp',focus='targcore',zoom=1.8)
    assert isinstance(fig, IPython.display.Video)

@pytest.mark.parametrize('ptype', ['mat','P','rho','phase'])
def test_snapshot_plot(snasphot_class_loaded, ptype):
    fig = snasphot_class_loaded.plot(ptype=ptype)
    assert isinstance(fig, matplotlib.pyplot.Figure)

