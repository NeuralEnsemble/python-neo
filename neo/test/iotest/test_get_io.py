from pathlib import Path
from tempfile import TemporaryDirectory
import platform
from neo.io import get_io, list_candidate_ios, NixIO
import pytest

try:
    import nixio

    HAVE_NIX = True
except:
    HAVE_NIX = False


def test_list_candidate_ios_non_existant_file():
    # use plexon io suffix for testing here
    non_existant_file = Path("non_existant_folder/non_existant_file.plx")
    non_existant_file.unlink(missing_ok=True)
    ios = list_candidate_ios(non_existant_file)

    assert ios

    # cleanup
    non_existant_file.unlink(missing_ok=True)


def test_list_candidate_ios_filename_stub():
    # create dummy folder with dummy files
    with TemporaryDirectory(prefix="filename_stub_test_") as test_folder:
        test_folder = Path(test_folder)
        test_filename = test_folder / "dummy_file.nix"
        test_filename.touch()
        filename_stub = test_filename.with_suffix("")

        # check that io is found even though file suffix was not provided
        ios = list_candidate_ios(filename_stub)

        assert NixIO in ios


@pytest.mark.skipif(not HAVE_NIX or platform.system() == "Windows", reason="Need nixio in order to return NixIO class")
def test_get_io_non_existant_file_writable_io():
    # use nixio for testing with writable io
    non_existant_file = Path("non_existant_file.nix")
    non_existant_file.unlink(missing_ok=True)
    io = get_io(non_existant_file)

    assert isinstance(io, NixIO)

    # cleanup
    non_existant_file.unlink(missing_ok=True)  # cleanup will fail on Windows so need to skip
