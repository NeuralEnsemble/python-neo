# -*- coding: utf-8 -*-
"""
Tests of neo.io.elphyio
"""

# needed for python 3 compatibility
from __future__ import division

import sys

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import ElphyIO
from neo.test.iotest.common_io_test import BaseTestIO


@unittest.skipIf(sys.version_info[0] > 2, "not Python 3 compatible")
class TestElphyIO(BaseTestIO, unittest.TestCase):
    ioclass = ElphyIO
    files_to_test = ['ElphyExample.DAT',
                     'ElphyExample_Mode1.dat',
                     'ElphyExample_Mode2.dat',
<<<<<<< HEAD
                     #'ElphyExample_Mode3.dat', # corrupted waveforms in this file
                     'DATA1.DAT',
                    ]
=======
                     'ElphyExample_Mode3.dat',
                     ]
>>>>>>> 3793a708d6bb55599ab731249dc777dac211b094
    files_to_download = files_to_test


if __name__ == "__main__":
    unittest.main()
