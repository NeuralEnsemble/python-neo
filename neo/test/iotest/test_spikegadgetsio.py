import unittest

from neo.io import SpikeGadgetsIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestSpikeGadgetsIO(BaseTestIO, unittest.TestCase, ):
    ioclass = SpikeGadgetsIO
    files_to_test =  ['20210225_em8_minirec2_ac.rec']
    files_to_download = files_to_test


if __name__ == "__main__":
    unittest.main()
