"""
Tests of neo.io.alphaomegaio
"""

import unittest

from neo.io import AlphaOmegaIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestAlphaOmegaIO(BaseTestIO, unittest.TestCase):
    entities_to_download =[
        'alphaomega'
    ]
    entities_to_test = [
        'alphaomega/mpx_map_version4',
    ]
    ioclass = AlphaOmegaIO


if __name__ == "__main__":
    unittest.main()
