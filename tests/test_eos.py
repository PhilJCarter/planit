import pytest
import random
import hashlib
import numpy as npy
from pathlib import Path
from planit import eos
from planit.eos import eosfuncs


CUSTOM_SESAME_DIR = Path(eos.eospath) / '5-phase-water'


def _synthetic_gadget_data():
    """Return a small Gadget table with visibly different rows and columns."""
    rho = npy.array([0.5, 1.0, 2.0, 4.0])
    entropy = npy.array([1.0e7, 2.0e7, 4.0e7])
    entropy_grid, density_grid = npy.meshgrid(entropy, rho, indexing='ij')

    # These values are in the units stored in a Gadget table.  The different
    # coefficients make a transposed or incorrectly converted array obvious.
    pressure = 1.0e9 + 2.0e8*density_grid + 3.0*entropy_grid
    temperature = 100.0 + 10.0*density_grid + entropy_grid/1.0e6
    internal_energy = 2.0e10 + 5.0e9*density_grid + 20.0*entropy_grid
    sound_speed = 1.0e5 + 1.0e4*density_grid + entropy_grid/1.0e3

    return {
        'rho': rho,
        'entropy': entropy,
        'pressure': pressure,
        'temperature': temperature,
        'internal_energy': internal_energy,
        'sound_speed': sound_speed,
    }


def _write_gadget_table(path, data=None, dimensions_scientific=False,
                        omit_values=0, extra_values=None):
    """Write one synthetic table in Gadget's serial ASCII layout."""
    if data is None:
        data = _synthetic_gadget_data()

    density = npy.asarray(data['rho'])
    entropy = npy.asarray(data['entropy'])
    arrays = [
        npy.asarray(data['pressure']),
        npy.asarray(data['temperature']),
        npy.asarray(data['internal_energy']),
        npy.asarray(data['sound_speed']),
    ]
    dimensions = (density.size, entropy.size)
    if dimensions_scientific:
        dimension_tokens = [f'{value:.8e}' for value in dimensions]
    else:
        dimension_tokens = [str(value) for value in dimensions]

    values = [*density, *entropy]
    for array in arrays:
        values.extend(array.ravel(order='C'))
    if omit_values:
        values = values[:-omit_values]
    if extra_values is not None:
        values.extend(extra_values)

    # Deliberately mix spaces, tabs, and line endings.  Gadget reads all of
    # these as whitespace, so a custom loader should do the same.
    value_tokens = [f'{value:.16e}' for value in values]
    tokens = dimension_tokens + value_tokens
    lines = []
    for index in range(0, len(tokens), 5):
        separator = '\t' if (index//5) % 2 else '  '
        lines.append(separator.join(tokens[index:index + 5]))
    path.write_text('\r\n'.join(lines) + '\r\n')
    return data


def _linear_bilinear(x, y, x_bounds, y_bounds, values):
    """Reference bilinear interpolation for one two-by-two cell."""
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    tx = (x - x0)/(x1 - x0)
    ty = (y - y0)/(y1 - y0)
    lower = values[0, 0]*(1.0 - tx) + values[0, 1]*tx
    upper = values[1, 0]*(1.0 - tx) + values[1, 1]*tx
    return lower*(1.0 - ty) + upper*ty


def _log_bilinear(x, y, x_bounds, y_bounds, values):
    """Reference log-bilinear interpolation for one two-by-two cell."""
    return 10.0**_linear_bilinear(
        npy.log10(x),
        npy.log10(y),
        npy.log10(npy.asarray(x_bounds)),
        npy.log10(npy.asarray(y_bounds)),
        npy.log10(npy.asarray(values)),
    )


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


def test_custom_gadget_table_parser_preserves_orientation_and_units(
        empty_user_eos_slots, tmp_path):
    table_path = tmp_path / 'synthetic-gadget.txt'
    data = _write_gadget_table(table_path, dimensions_scientific=True)

    table = eos.select(
        'User0',
        eosname='Synthetic Gadget material',
        gadget_file=table_path,
    )

    assert table.TYPE == 'GADGET'
    assert table.MODELNAME == 'Synthetic Gadget material'
    assert table.womaID == 900
    assert table.source_path == str(table_path.resolve())
    assert table.source_size == table_path.stat().st_size
    assert table.source_sha256 == hashlib.sha256(table_path.read_bytes()).hexdigest()
    assert table.ND == len(data['rho'])
    assert table.NS == len(data['entropy'])
    npy.testing.assert_allclose(table.rho, data['rho'])
    npy.testing.assert_allclose(table.S, data['entropy']/1.0e10)
    npy.testing.assert_allclose(table.P, data['pressure']/1.0e10)
    npy.testing.assert_allclose(table.T, data['temperature'])
    npy.testing.assert_allclose(table.U, data['internal_energy']/1.0e10)
    npy.testing.assert_allclose(table.cs, data['sound_speed'])

    # Check the public cgs interface at two corners as well as the table
    # object, including the last row and column.
    rho = npy.array([data['rho'][0], data['rho'][-1]])
    entropy = npy.array([data['entropy'][0], data['entropy'][-1]])
    mats = npy.array([900, 900])
    indices = ((0, 0), (-1, -1))
    for label, key in (
            ('P', 'pressure'),
            ('T', 'temperature'),
            ('U', 'internal_energy'),
            ('cs', 'sound_speed')):
        expected = npy.array([data[key][index] for index in indices])
        actual = eos.calcprop(label, 'rho', 'S', rho, entropy, mats)
        npy.testing.assert_allclose(actual, expected, rtol=1.0e-12)


def test_custom_gadget_table_uses_linear_interpolation_when_requested(
        empty_user_eos_slots, tmp_path):
    table_path = tmp_path / 'linear-gadget.txt'
    data = _write_gadget_table(table_path)
    eos.select(
        'User0',
        eosname='Linear Gadget material',
        gadget_file=table_path,
        gadget_low_density_log=False,
    )

    rho = 0.75
    entropy = 1.5e7
    expected = _linear_bilinear(
        rho,
        entropy,
        data['rho'][:2],
        data['entropy'][:2],
        data['internal_energy'][:2, :2],
    )
    actual = eos.calcprop(
        'U', 'rho', 'S',
        npy.array([rho]), npy.array([entropy]), npy.array([900]),
    )
    assert actual[0] == pytest.approx(expected, rel=1.0e-12)


def test_custom_gadget_table_can_use_low_density_log_interpolation(
        empty_user_eos_slots, tmp_path):
    table_path = tmp_path / 'log-gadget.txt'
    data = _write_gadget_table(table_path)
    eos.select(
        'User0',
        eosname='Log Gadget material',
        gadget_file=table_path,
        gadget_low_density_log=True,
    )

    rho = npy.sqrt(data['rho'][0]*data['rho'][1])
    entropy = npy.sqrt(data['entropy'][0]*data['entropy'][1])
    expected = _log_bilinear(
        rho,
        entropy,
        data['rho'][:2],
        data['entropy'][:2],
        data['internal_energy'][:2, :2],
    )
    actual = eos.calcprop(
        'U', 'rho', 'S',
        npy.array([rho]), npy.array([entropy]), npy.array([900]),
    )
    assert actual[0] == pytest.approx(expected, rel=1.0e-12)


def test_custom_gadget_log_interpolation_rejects_nonpositive_values(
        empty_user_eos_slots, tmp_path):
    data = _synthetic_gadget_data()
    data['pressure'][0, 0] = 0.0
    table_path = tmp_path / 'nonpositive-log-gadget.txt'
    _write_gadget_table(table_path, data=data)
    eos.select(
        'User0',
        eosname='Non-positive log Gadget material',
        gadget_file=table_path,
        gadget_low_density_log=True,
    )

    with pytest.raises(ValueError, match='requires positive entropy and property'):
        eos.calcprop(
            'P', 'rho', 'S',
            npy.array([0.75]), npy.array([1.5e7]), npy.array([900]),
        )


def test_custom_gadget_linear_interpolation_accepts_nonpositive_entropy(
        empty_user_eos_slots, tmp_path):
    data = _synthetic_gadget_data()
    data['entropy'] = npy.array([-1.0e7, 0.0, 1.0e7])
    table_path = tmp_path / 'linear-entropy-zero-gadget.txt'
    _write_gadget_table(table_path, data=data)
    eos.select(
        'User0',
        eosname='Linear entropy-zero Gadget material',
        gadget_file=table_path,
    )

    rho = 0.75
    entropy = -0.5e7
    expected = _linear_bilinear(
        rho,
        entropy,
        data['rho'][:2],
        data['entropy'][:2],
        data['internal_energy'][:2, :2],
    )
    actual = eos.calcprop(
        'U', 'rho', 'S',
        npy.array([rho]), npy.array([entropy]), npy.array([900]),
    )
    assert actual[0] == pytest.approx(expected, rel=1.0e-12)

    eos.select(
        'User0',
        eosname='Log entropy-zero Gadget material',
        gadget_file=table_path,
        gadget_low_density_log=True,
    )
    with pytest.raises(ValueError, match='requires positive entropy and property'):
        eos.calcprop(
            'U', 'rho', 'S',
            npy.array([rho]), npy.array([entropy]), npy.array([900]),
        )


def test_custom_gadget_log_interpolation_switches_above_two_g_per_cc(
        empty_user_eos_slots, tmp_path):
    data = _synthetic_gadget_data()
    table_path = tmp_path / 'threshold-gadget.txt'
    _write_gadget_table(table_path, data=data)
    eos.select(
        'User0',
        eosname='Threshold Gadget material',
        gadget_file=table_path,
        gadget_low_density_log=True,
    )

    entropy = npy.sqrt(data['entropy'][1]*data['entropy'][2])
    rho_at_threshold = 2.0
    rho_above_threshold = npy.nextafter(2.0, npy.inf)
    cell = data['internal_energy'][1:3, 2:4]
    expected_at_threshold = _log_bilinear(
        rho_at_threshold,
        entropy,
        data['rho'][2:4],
        data['entropy'][1:3],
        cell,
    )
    expected_above_threshold = _linear_bilinear(
        rho_above_threshold,
        entropy,
        data['rho'][2:4],
        data['entropy'][1:3],
        cell,
    )
    actual = eos.calcprop(
        'U', 'rho', 'S',
        npy.array([rho_at_threshold, rho_above_threshold]),
        npy.array([entropy, entropy]),
        npy.array([900, 900]),
    )
    npy.testing.assert_allclose(
        actual,
        [expected_at_threshold, expected_above_threshold],
        rtol=1.0e-12,
    )


def test_custom_gadget_table_bounds_error_and_clip(
        empty_user_eos_slots, tmp_path):
    table_path = tmp_path / 'bounded-gadget.txt'
    data = _write_gadget_table(table_path)
    eos.select(
        'User0',
        eosname='Strict Gadget material',
        gadget_file=table_path,
        gadget_out_of_domain='error',
    )

    with pytest.raises(ValueError, match='(?i)(outside|range|domain)'):
        eos.calcprop(
            'U', 'rho', 'S',
            npy.array([data['rho'][0]/2.0]),
            npy.array([data['entropy'][-1]*2.0]),
            npy.array([900]),
        )

    eos.select(
        'User0',
        eosname='Clipped Gadget material',
        gadget_file=table_path,
        gadget_out_of_domain='clip',
    )
    actual = eos.calcprop(
        'U', 'rho', 'S',
        npy.array([data['rho'][0]/2.0]),
        npy.array([data['entropy'][-1]*2.0]),
        npy.array([900]),
    )
    assert actual[0] == pytest.approx(data['internal_energy'][-1, 0])


def test_custom_gadget_table_allows_negative_pressure_at_high_density(
        empty_user_eos_slots, tmp_path):
    data = _synthetic_gadget_data()
    data['pressure'] = npy.array([
        [-100.0, -80.0, -60.0, -40.0],
        [-90.0, -70.0, -50.0, -30.0],
        [-70.0, -50.0, -30.0, -10.0],
    ])
    table_path = tmp_path / 'negative-pressure-gadget.txt'
    _write_gadget_table(table_path, data=data)
    eos.select(
        'User0',
        eosname='Negative-pressure Gadget material',
        gadget_file=table_path,
        gadget_low_density_log=True,
    )

    rho = 3.0
    entropy = 3.0e7
    expected = _linear_bilinear(
        rho,
        entropy,
        data['rho'][2:4],
        data['entropy'][1:3],
        data['pressure'][1:3, 2:4],
    )
    actual = eos.calcprop(
        'P', 'rho', 'S',
        npy.array([rho]), npy.array([entropy]), npy.array([900]),
    )
    assert actual[0] == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)


@pytest.mark.parametrize(
    'malformation',
    ['truncated', 'extra', 'nonmonotonic-density', 'nonfinite-entropy'],
)
def test_custom_gadget_table_rejects_malformed_files(
        empty_user_eos_slots, tmp_path, malformation):
    table_path = tmp_path / f'{malformation}.txt'
    data = _synthetic_gadget_data()
    if malformation == 'truncated':
        _write_gadget_table(table_path, data=data, omit_values=1)
    elif malformation == 'extra':
        _write_gadget_table(table_path, data=data, extra_values=[123.0])
    elif malformation == 'nonmonotonic-density':
        data['rho'] = npy.array([0.5, 2.0, 1.0, 4.0])
        _write_gadget_table(table_path, data=data)
    else:
        data['entropy'] = npy.array([1.0e7, npy.nan, 4.0e7])
        _write_gadget_table(table_path, data=data)

    with pytest.raises(ValueError):
        eos.select(
            'User0',
            eosname='Malformed Gadget material',
            gadget_file=table_path,
        )


def test_failed_custom_gadget_replacement_preserves_loaded_slot(
        empty_user_eos_slots, tmp_path):
    valid_path = tmp_path / 'valid-gadget.txt'
    malformed_path = tmp_path / 'malformed-gadget.txt'
    _write_gadget_table(valid_path)
    _write_gadget_table(malformed_path, omit_values=1)
    first = eos.select(
        'User0',
        eosname='Valid Gadget material',
        gadget_file=valid_path,
    )

    with pytest.raises(ValueError):
        eos.select(
            'User0',
            eosname='Malformed replacement',
            gadget_file=malformed_path,
        )
    assert eos.select('User0') is first


def test_custom_gadget_slots_support_mixed_material_vectors_and_scalars(
        empty_user_eos_slots, tmp_path):
    first_data = _synthetic_gadget_data()
    second_data = {
        key: npy.array(value, copy=True)
        for key, value in first_data.items()
    }
    second_data['internal_energy'] *= 2.0
    first_path = tmp_path / 'first-gadget.txt'
    second_path = tmp_path / 'second-gadget.txt'
    _write_gadget_table(first_path, data=first_data)
    _write_gadget_table(second_path, data=second_data)
    eos.select(
        'User0', eosname='First Gadget material', gadget_file=first_path,
    )
    eos.select(
        'User1', eosname='Second Gadget material', gadget_file=second_path,
    )

    rho = npy.array([first_data['rho'][1], second_data['rho'][2]])
    entropy = npy.array([
        first_data['entropy'][1], second_data['entropy'][2],
    ])
    actual = eos.calcprop(
        'U', 'rho', 'S', rho, entropy, npy.array([900, 901]),
    )
    expected = npy.array([
        first_data['internal_energy'][1, 1],
        second_data['internal_energy'][2, 2],
    ])
    npy.testing.assert_allclose(actual, expected, rtol=1.0e-12)

    scalar = eos.calcprop(
        'U', 'rho', 'S',
        first_data['rho'][1], first_data['entropy'][1], 900,
    )
    assert npy.asarray(scalar).reshape(-1)[0] == pytest.approx(expected[0])


def test_custom_gadget_source_is_explicit_and_sesame_syntax_is_unchanged(
        empty_user_eos_slots, tmp_path):
    table_path = tmp_path / 'synthetic-gadget.txt'
    _write_gadget_table(table_path)

    with pytest.raises(ValueError):
        eos.select(
            'User0',
            eosname='Ambiguous material',
            eosdir=CUSTOM_SESAME_DIR,
            gadget_file=table_path,
        )

    sesame = eos.select(
        'User0',
        eosname='Custom SESAME material',
        eosdir=CUSTOM_SESAME_DIR,
    )
    assert sesame.TYPE == 'SESAME'
    assert sesame.womaID == 900

    with pytest.raises(ValueError, match='only be used with User0 through User4'):
        eos.select('ANEOSIron', gadget_file=table_path, eosname='Wrong slot')


def test_calcprop_accepts_empty_arrays():
    result = eos.calcprop(
        'U', 'rho', 'S',
        npy.array([]), npy.array([]), npy.array([], dtype=int),
    )
    assert result.shape == (0,)
    assert result.dtype == float

    with pytest.raises(NotImplementedError):
        eos.calcprop(
            'not-a-property', 'rho', 'S',
            npy.array([]), npy.array([]), npy.array([], dtype=int),
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
