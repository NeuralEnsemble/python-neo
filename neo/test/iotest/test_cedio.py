import unittest
from platform import system
from sys import maxsize

try:
    if system() == "Windows":
        if maxsize > 2**32:
            import sonpy.amd64.sonpy
        else:
            import sonpy.win32.sonpy
    elif system() == "Darwin":
        import sonpy.darwin.sonpy
    elif system() == "Linux":
        import sonpy.linux.sonpy
    from neo.io import CedIO
except ImportError:
    HAVE_SONPY = False
    CedIO = None
else:
    HAVE_SONPY = True

from neo.test.iotest.common_io_test import BaseTestIO


@unittest.skipUnless(HAVE_SONPY, "sonpy")
class TestCedIO(
    BaseTestIO,
    unittest.TestCase,
):
    ioclass = CedIO
    entities_to_test = ["spike2/m365_1sec.smrx", "spike2/File_spike2_1.smr", "spike2/Two-mice-bigfile-test000.smr"]
    entities_to_download = ["spike2"]


if __name__ == "__main__":
    unittest.main()
