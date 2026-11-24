import unittest

try:
    from neo.rawio.cedrawio import _get_sonpy_namespace

    # Raises ImportError if sonpy is missing or exposes no usable namespace.
    _get_sonpy_namespace()
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
