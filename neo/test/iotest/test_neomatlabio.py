# encoding: utf-8
"""
Tests of io.asciisignalio
"""
from __future__ import absolute_import, division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

try:
    from ...io import NeoMatlabIO
    can_run = True
except ImportError:
    can_run = False
    NeoMatlabIO = None
    
import numpy

from .common_io_test import BaseTestIO

@unittest.skipUnless(can_run, "NeoMatlabIO not available")
class TestNeoMatlabIO(BaseTestIO, unittest.TestCase):
    ioclass = NeoMatlabIO
    files_to_test = [ ]
    files_to_download = [ ]


if __name__ == "__main__":
    unittest.main()
