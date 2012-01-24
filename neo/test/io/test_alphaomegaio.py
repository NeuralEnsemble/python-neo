# encoding: utf-8
"""
Tests of io.alphaomegaio
"""

from __future__ import division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import AlphaOmegaIO

from neo.test.io.common_io_test import BaseTestIO


class TestAlphaOmegaIO(BaseTestIO, unittest.TestCase):
    files_to_test = ['File_AlphaOmega_1.map',
                     'File_AlphaOmega_2.map']
    files_to_download = files_to_test
    ioclass = AlphaOmegaIO


if __name__ == "__main__":
    unittest.main()
