import unittest

from neo.io import SpikeGadgetsIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestSpikeGadgetsIO(BaseTestIO, unittest.TestCase, ):
    ioclass = SpikeGadgetsIO
    entities_to_download = ['spikegadgets']
    entities_to_test = [
        'spikegadgets/20210225_em8_minirec2_ac.rec',
        'spikegadgets/W122_06_09_2019_1_fromSD.rec'
    ]


if __name__ == "__main__":
    unittest.main()
