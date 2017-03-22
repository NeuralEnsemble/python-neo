# -*- coding: utf-8 -*-
"""
Tests of neo.io.alphaomegaio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import AlphaOmegaIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestAlphaOmegaIO(BaseTestIO, unittest.TestCase):
    files_to_test = ['File_AlphaOmega_1.map',
                     'File_AlphaOmega_2.map']
    files_to_download = files_to_test
    ioclass = AlphaOmegaIO


if __name__ == "__main__":
    unittest.main()
