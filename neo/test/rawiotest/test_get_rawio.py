from neo.rawio import get_rawio
from pathlib import Path
from tempfile import TemporaryDirectory


def test_get_rawio_class():
    # use plexon io suffix for testing here
    non_existant_file = Path("non_existant_folder/non_existant_file.plx")
    non_existant_file.unlink(missing_ok=True)
    ios = get_rawio(non_existant_file)

    assert ios

    # cleanup
    non_existant_file.unlink(missing_ok=True)


def test_get_rawio_class_nonsupported_rawio():

    non_existant_file = Path("non_existant_folder/non_existant_file.fake")
    non_existant_file.unlink(missing_ok=True)
    ios = get_rawio(non_existant_file)

    assert ios is None

    # cleanup
    non_existant_file.unlink(missing_ok=True)
