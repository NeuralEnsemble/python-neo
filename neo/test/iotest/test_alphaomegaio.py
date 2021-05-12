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
        'alphaomega/File_AlphaOmega_1.map',
        'alphaomega/File_AlphaOmega_2.map'
    ]
    ioclass = AlphaOmegaIO


if __name__ == "__main__":
    unittest.main()
