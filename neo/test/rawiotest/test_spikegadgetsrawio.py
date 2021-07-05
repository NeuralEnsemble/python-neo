import unittest

from neo.rawio.spikegadgetsrawio import SpikeGadgetsRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestSpikeGadgetsRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = SpikeGadgetsRawIO
    entities_to_download = ['spikegadgets']
    entities_to_test = [
        'spikegadgets/20210225_em8_minirec2_ac.rec',
        'spikegadgets/W122_06_09_2019_1_fromSD.rec'
    ]


if __name__ == "__main__":
    unittest.main()
