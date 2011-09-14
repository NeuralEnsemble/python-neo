# encoding: utf-8
"""
Tests of io.NeuroExplorerIO
"""

from __future__ import division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import NeuroExplorerIO
import numpy

from neo.test.io.common_io_test import BaseTestIO



class TestNeuroExplorerIO(BaseTestIO, unittest.TestCase, ):
    ioclass = NeuroExplorerIO
    files_to_test = [ 'File_neuroexplorer_1.nex',
                            'File_neuroexplorer_2.nex',
                            ]
    files_to_download = files_to_test



if __name__ == "__main__":
    unittest.main()
