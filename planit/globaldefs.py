import os

G = 6.6743E-8     # Gravitational constant cgs cm3/g/s2
Mearth = 5.972E27 # Earth's mass g
Rearth = 6.371e8  # Earth's radius in cm

GADGET_EOS_OFFSET = IDOFF   = 200000000    # material id offset
PROJ_ID_OFFSET    = BODYOFF = 100000000    # body id offset

eospath = os.path.dirname(__file__) + '/eos/data/'
datadir = os.path.dirname(__file__) + '/data/'
