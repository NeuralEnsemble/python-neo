import unittest

from neo.rawio.cedrawio import CedRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO

try:
    import sonpy

    HAVE_SONPY = True
except ImportError:
    HAVE_SONPY = False


# This runs standard tests, this is mandatory for all IOs
@unittest.skipUnless(HAVE_SONPY, "requires sonpy package and all its dependencies")
class TestCedRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = CedRawIO
    entities_to_test = ["spike2/m365_1sec.smrx", "spike2/File_spike2_1.smr", "spike2/Two-mice-bigfile-test000.smr"]
    entities_to_download = ["spike2"]


if __name__ == "__main__":
    unittest.main()
