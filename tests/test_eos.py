import pytest
import random
import numpy as npy
from pathlib import Path
from planit import eos
from planit.eos import eosfuncs


CUSTOM_SESAME_DIR = Path(eos.eospath) / '5-phase-water'


@pytest.fixture
def empty_user_eos_slots(monkeypatch):
    """Isolate the module-level custom-EOS cache for each focused test."""
    for index in range(5):
        monkeypatch.setattr(eosfuncs, f'UserEOS{index}', None)


@pytest.mark.parametrize(
    ('EOS', 'womaID'),
    [
        ('iron', 401),
        ('Fe', 401),
        ('FeSi', 402),
        ('Fo', 400),
        ('ANEOSPyrolite', 403),
        ('5PhaseWater', 303),
    ],
)
def test_eos_loading(EOS, womaID):
    table = eos.select(EOS)
    assert table is not None
    assert table.womaID == womaID

def test_unknown_eos_loading():
    with pytest.raises(ValueError):
        eos.select('Cheese')


def test_user_eos_slots_load_with_ids_and_interpolate(empty_user_eos_slots):
    """All user slots accept SESAME data without mutating bundled tables."""
    for index in range(5):
        slot = f'User{index}'
        womaID = 900 + index
        eosdir = str(CUSTOM_SESAME_DIR)
        if index % 2:
            eosdir += '/'

        selector = womaID if index == 4 else slot
        table = eos.select(selector, eosname=f'Custom material {index}', eosdir=eosdir)

        assert table.womaID == womaID
        assert table.MODELNAME == f'Custom material {index}'
        assert table.TYPE == 'SESAME'
        assert eos.select(womaID) is table

    table = eos.select('User0')
    density_index = len(table.rho) // 2
    temperature_index = len(table.T) // 2
    passer = table.make_passer_class()
    assert eos.tabinterp.from_rhoT(
        'U',
        table.rho[density_index] * (1.0 + 1e-8),
        table.T[temperature_index] * (1.0 + 1e-12),
        passer,
    ) == pytest.approx(table.U[temperature_index, density_index], rel=1e-3, abs=1e-11)


def test_user_eos_replacement_is_explicit_and_failed_load_preserves_cache(
        empty_user_eos_slots, tmp_path):
    first = eos.select('User0', eosname='First material', eosdir=CUSTOM_SESAME_DIR)
    replacement = eos.select('User0', eosname='Replacement material', eosdir=CUSTOM_SESAME_DIR)

    assert replacement is not first
    assert replacement.MODELNAME == 'Replacement material'
    assert eos.select('User0') is replacement

    missing_dir = tmp_path / 'does-not-exist'
    with pytest.raises(FileNotFoundError, match='directory does not exist'):
        eos.select('User0', eosname='Broken replacement', eosdir=missing_dir)
    assert eos.select('User0') is replacement


@pytest.mark.parametrize(
    'kwargs',
    [
        {'eosname': 'Only a name'},
        {'eosdir': CUSTOM_SESAME_DIR},
    ],
)
def test_user_eos_rejects_incomplete_request(empty_user_eos_slots, kwargs):
    with pytest.raises(ValueError, match='requires both eosname and eosdir'):
        eos.select('User0', **kwargs)


@pytest.mark.parametrize('eosname', ['', 42])
def test_user_eos_validates_eosname(empty_user_eos_slots, eosname):
    with pytest.raises(ValueError, match='eosname must be a non-empty string'):
        eos.select('User0', eosname=eosname, eosdir=CUSTOM_SESAME_DIR)


def test_user_eos_validates_unloaded_slot_and_slot_name(empty_user_eos_slots):
    with pytest.raises(ValueError, match='is not loaded'):
        eos.select('User0')
    with pytest.raises(ValueError, match='Unknown user EOS slot'):
        eos.select('User5', eosname='Out of range', eosdir=CUSTOM_SESAME_DIR)


def test_user_eos_validates_missing_directory(empty_user_eos_slots, tmp_path):
    missing_dir = tmp_path / 'does-not-exist'
    with pytest.raises(FileNotFoundError, match='directory does not exist'):
        eos.select('User0', eosname='Missing directory', eosdir=missing_dir)


def test_user_eos_validates_non_directory_path(empty_user_eos_slots, tmp_path):
    file_path = tmp_path / 'not-a-directory'
    file_path.write_text('not an EOS directory')

    with pytest.raises(NotADirectoryError, match='path is not a directory'):
        eos.select('User0', eosname='Not a directory', eosdir=file_path)


def test_user_eos_validates_required_files(empty_user_eos_slots, tmp_path):
    incomplete_dir = tmp_path / 'incomplete-sesame'
    incomplete_dir.mkdir()

    with pytest.raises(FileNotFoundError, match='NEW-SESAME-STD.TXT'):
        eos.select('User0', eosname='Incomplete table', eosdir=incomplete_dir)


def test_direct_user_loader_requires_a_reserved_woma_id():
    with pytest.raises(ValueError, match='womaID must be one of'):
        eos.loadANEOSEOS(
            eos='Direct custom table',
            eostype='SESAME',
            eosdir=CUSTOM_SESAME_DIR,
            user=True,
        )

def test_isentrope_init():
    i = eos.isentrope_class()
    assert i.entropy is None

def test_calcprop_unknown():
    with pytest.raises(Exception):
        eos.calcprop('3', 'rho', 'T', 4, 3000, 401)


@pytest.mark.parametrize('execcount', range(1000))
def test_interp_ANEOS_U(execcount):
    aneoslist = ['ANEOSIron', 'ANEOSForsterite', 'ANEOSFeSiAlloy', 'ANEOSPyrolite', '5PhaseWater']
    EOS = eos.select(random.choice(aneoslist))
    j = npy.random.randint(0, high=len(EOS.rho))
    i = npy.random.randint(0, high=len(EOS.T))
    print(EOS.MODELNAME, j, i)
    EOSpasser = EOS.make_passer_class()
    assert eos.tabinterp.from_rhoT('U', EOS.rho[j]*(1.+1e-8), EOS.T[i]*(1.+1e-12), EOSpasser) == pytest.approx(EOS.U[i,j], rel=1e-3, abs=1e-11)

@pytest.mark.parametrize('execcount', range(1000))
def test_interp_ANEOS_S(execcount):
    aneoslist = ['ANEOSIron', 'ANEOSForsterite', 'ANEOSFeSiAlloy', '5PhaseWater']
    EOS = eos.select(random.choice(aneoslist))
    j = npy.random.randint(2, high=len(EOS.rho))
    i = npy.random.randint(12, high=len(EOS.T))
    print(EOS.MODELNAME, j, i)
    EOSpasser = EOS.make_passer_class()
    #U = eos.tabinterp.from_rhoT('U',EOSpasser.rho[j]*(1.-1e-15),EOSpasser.T[i]*(1.-1e-15),EOSpasser)
    assert eos.tabinterp.from_rhoU('S', EOS.rho[j]*(1.-1e-15), EOS.U[i,j]*(1.-1e-15), EOSpasser) == pytest.approx(EOS.S[i,j], rel=1.2e-1, abs=1e-6)
