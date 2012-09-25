# encoding: utf-8
"""
Tests of io.alphaomegaio
"""
from __future__ import absolute_import, division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from ...io import AlphaOmegaIO

from .common_io_test import BaseTestIO


class TestAlphaOmegaIO(BaseTestIO, unittest.TestCase):
    files_to_test = ['File_AlphaOmega_1.map',
                     'File_AlphaOmega_2.map']
    files_to_download = files_to_test
    ioclass = AlphaOmegaIO


if __name__ == "__main__":
    unittest.main()
