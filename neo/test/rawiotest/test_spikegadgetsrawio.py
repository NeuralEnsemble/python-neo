import unittest

from neo.rawio.spikegadgetsrawio import SpikeGadgetsRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestSpikeGadgetsRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = SpikeGadgetsRawIO
    entities_to_test = ['20210225_em8_minirec2_ac.rec']
    files_to_download = entities_to_test


if __name__ == "__main__":
    unittest.main()
