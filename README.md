# planit

[![Tests](https://github.com/PhilJCarter/planit/actions/workflows/python-test.yml/badge.svg)](https://github.com/PhilJCarter/planit/actions/workflows/python-test.yml)

The **Plan**etary **I**mpact **T**oolkit provides tools for accessing, manipulating, and 
analysing smoothed particle hydrodynamcis (SPH) impact simulation data, and creating 
planets for SPH impact simulations.

Philip J. Carter (p.carter@bristol.ac.uk)\
Sarah T. Stewart (sstewa56@asu.edu)

## Overview

**planit** currently supports the SPH codes [Gadget2-planetary](https://github.com/PlanetSim/gadget2-planetary) and [SWIFT](http://www.swiftsim.com/).

`makeplanet` provides functions for generating adiabatic 1D planet structures and 
converting these into (unequilibrated) SPH planets.\
`snaptools` provides the `Snapshot` class for holding input and output files, calculating 
remnant masses, and vapor fractions.\
`impacttools` provides the `Impact` class for loading and plotting snapshots produced by 
an impact simulation.\
`eos` provides classes and functions for accessing and using equations of state.\
`utils` provides additional utility functions for analysis.

> [!NOTE]
> Currently only two-layer planets with a forsterite mantle are *fully* supported.

## Installation
**planit** is available in PyPI and can be installed with `pip install planit`.

## Equations of State (EoS)

**planit** currently includes the following EoS:
- ANEOS Forsterite ([Stewart et al., 2019](https://github.com/ststewart/aneos-forsterite-2019)): `ANEOSForsterite`
- ANEOS Iron ([Stewart, 2020](https://github.com/ststewart/aneos-iron-2020)): `ANEOSIron`
- ANEOS Fe85Si15 ([Stewart, 2020](https://github.com/ststewart/aneos-Fe85Si15-2020/)): `ANEOSFeSiAlloy`
- ANEOS Pyrolite ([Stewart et al., 2022](https://github.com/ststewart/aneos-pyrolite-2022/)): `ANEOSPyrolite`
- 5-phase Water ([Senft & Stewart, 2008](https://ui.adsabs.harvard.edu/link_gateway/2008M&PS...43.1993S/doi:10.1111/j.1945-5100.2008.tb00657.x)): `5PhaseWater`
- HM80 Hydrogen-Helium ([Hubbard & MacFarlane, 1980](https://ui.adsabs.harvard.edu/link_gateway/1980JGR....85..225H/doi:10.1029/JB085iB01p00225); [Lock & Stewart, 2024](https://ui.adsabs.harvard.edu/link_gateway/2024PSJ.....5...28L/doi:10.3847/PSJ/ad0b16)): `HM80HHe`

They can be accessed using the names given above (as well as some variations) or using the 
corresponding SWIFT/WoMa ID number.\
User supplied SESAME tables can also be loaded into one of five custom slots.

### Custom SESAME EoS tables

**planit** supports custom tables in the same SESAME-style layout used by its bundled ANEOS
and 5-phase-water tables. A table directory must contain all of these files:

- `NEW-SESAME-STD.TXT`
- `NEW-SESAME-STD-NOTENSION.TXT`
- `NEW-SESAME-EXT.TXT`

Use `User0` through `User4` to choose the corresponding reserved SWIFT/WoMa ID:
`User0` = 900, `User1` = 901, `User2` = 902, `User3` = 903, and `User4` = 904.
The IDs can also be passed to `select()` in place of the slot names.

```python
from planit import eos

table = eos.select(
    "User0",
    eosname="MyMaterial",
    eosdir="/path/to/my-sesame-table",  # trailing slash is optional
)

print(table.MODELNAME)  # MyMaterial
print(table.womaID)     # 900

# Use the loaded table in the usual way.
eos_passer = table.make_passer_class()
internal_energy = eos.tabinterp.from_rhoT("U", 1.0, 300.0, eos_passer)
```

The first load for a slot requires both `eosname` and `eosdir`. Subsequent
`eos.select("User0")` calls return that slot's cached table. Passing both arguments
again deliberately replaces the slot, but only after the new table has loaded
successfully; a failed replacement raises an error and leaves the previous cached table
available. Passing just one argument, or reading a slot that was never loaded, raises a
clear `ValueError` instead of returning an uninitialised table.

The loader validates the directory and file names, but it does not validate the physical
quality or full internal consistency of a table. `NEW-SESAME-EXT.TXT` is required, so a
minimal standard SESAME 201/301 file alone is not sufficient. Custom tables are cached
only in the current Python process, and there are only five simultaneous custom slots.

>[!WARNING]
> HM80 EoS support is not fully tested yet!

> [!NOTE]
> `ANEOSPyrolite` has been assigned the SWIFT/WoMa ID 403 in planit but does not currently have an 
assigned ID in SWIFT. Using a modified version of SWIFT or using the user supplied EoS functionality 
will be required.


## Planet generation

`planit.makeplanet` differs from other planet creation code in that produces planets with 
isentropic layers, integrates outwards from the centre of the planet, and ensures the 
requested mass and component mass fractions are matched (within tolerance). The default 
behaviour prevents the unphyiscal scenario of a core cooler than the mantle at the 
core-mantle boundary.

