# -*- coding: utf-8 -*-

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.rawio.neuroexplorerrawio import NeuroExplorerRawIO

from neo.rawio.tests.common_rawio_test import BaseTestRawIO


class TestNeuroExplorerRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = NeuroExplorerRawIO
    files_to_test = [
                    #~ 'File_neuroexplorer_1.nex',
                    'File_neuroexplorer_2.nex',
                     ]
    files_to_download = files_to_test

if __name__ == "__main__":
    unittest.main()

