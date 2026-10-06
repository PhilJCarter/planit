import pytest
import numpy as npy
import matplotlib
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
    fig = i.plotseq(n=3,type='phase')
    assert isinstance(fig, matplotlib.pyplot.Figure)


def test_impact_plotseq_entropy(tmp_reference_impact_seq):
    i = Impact()
    i.load(tmp_reference_impact_seq, prefix='snap', ndigits=3, thermo=True,compress=False)
    fig = i.plotseq(n=3,type='S',focus='targcore',scale='Earth',zoom=1.0)
    assert isinstance(fig, matplotlib.pyplot.Figure)


def test_impact_plotseq_density(tmp_reference_impact_seq):
    i = Impact()
    i.load(tmp_reference_impact_seq, prefix='snap', ndigits=3)
    fig = i.plotseq(n=3,type='rho',focus='potmin',scale='km',zoom=1.5)
    assert isinstance(fig, matplotlib.pyplot.Figure)



