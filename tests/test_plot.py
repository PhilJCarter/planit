import pytest
import numpy as npy
import matplotlib
matplotlib.use('Agg')
import IPython
from planit.globaldefs import *
from planit import Impact


def test_impact_plotseq_material(tmp_reference_impact_seq):
    i = Impact()
    i.load(tmp_reference_impact_seq, prefix='snap', ndigits=3)
    fig = i.plotseq(n=4,type='mats',zoom=1.2)
    assert isinstance(fig, matplotlib.pyplot.Figure)


def test_impact_plotseq_phase(tmp_reference_impact_seq):
    i = Impact()
    i.load(tmp_reference_impact_seq, prefix='snap', ndigits=3, thermo=True, compress=False)
    fig = i.plotseq(n=3,ptype='phase')
    assert isinstance(fig, matplotlib.pyplot.Figure)


def test_impact_plotseq_entropy(tmp_reference_impact_seq):
    i = Impact()
    i.load(tmp_reference_impact_seq, prefix='snap', ndigits=3, thermo=True,compress=False)
    fig = i.plotseq(n=2,type='S',focus='targcore',scale='Earth',zoom=1.0)
    assert isinstance(fig, matplotlib.pyplot.Figure)


def test_impact_plotseq_density(tmp_reference_impact_seq):
    i = Impact()
    i.load(tmp_reference_impact_seq, prefix='snap', ndigits=3)
    fig = i.plotseq(n=2,ptype='rho',focus='potmin',scale='km',zoom=1.5)
    assert isinstance(fig, matplotlib.pyplot.Figure)


def test_impact_mov_material_single(tmp_reference_impact_seq):
    i = Impact()
    i.load(tmp_reference_impact_seq, prefix='snap', ndigits=3)
    fig = i.attrmov(seq=[0,],ptype='mats',zoom=1.2)
    assert isinstance(fig, matplotlib.pyplot.Figure)


def test_impact_mov_pressure(tmp_reference_impact_seq):
    i = Impact()
    i.load(tmp_reference_impact_seq, prefix='snap', ndigits=3, thermo=True)
    fig = i.attrmov(seq=[0,2,6],ptype='P',zoom=1.8)
    assert isinstance(fig, IPython.display.Video)


def test_impact_mov_temperature(tmp_reference_impact_seq):
    i = Impact()
    i.load(tmp_reference_impact_seq, prefix='snap', ndigits=3, thermo=True)
    fig = i.attrmov(n=3,ptype='temp',focus='targcore',zoom=1.8)
    assert isinstance(fig, IPython.display.Video)



