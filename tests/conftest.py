import pytest
import os
import shutil

# __file__ is the absolute path to this current python script.
# .parent gets the directory containing this script (the 'tests' folder).
TEST_DIR = os.path.dirname(__file__)
DATA_DIR = TEST_DIR + '/testdata/'

@pytest.fixture
def reference_snapshot():
    """Returns the absolute path to the static reference HDF5 file."""
    file_path = DATA_DIR + 'reference_snapshot_100000.hdf5'
    assert os.path.exists(file_path)
    
    return file_path


@pytest.fixture
def tmp_reference_snapshot(tmp_path):
    """Creates a fresh copy of the data file for any test that requests it."""
    source_file = DATA_DIR + 'reference_snapshot_100000.hdf5'
    
    tmp_file = str(tmp_path) + 'temp_snapshot.hdf5'
    shutil.copy(source_file, tmp_file)
    assert os.path.exists(tmp_file)
    
    yield tmp_file