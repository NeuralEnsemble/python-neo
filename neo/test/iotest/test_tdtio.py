# -*- coding: utf-8 -*-
"""
Tests of neo.io.tdtio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

import sys

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import TdtIO
from neo.test.iotest.common_io_test import BaseTestIO


@unittest.skipIf(sys.version_info[0] > 2, "not Python 3 compatible")
class TestTdtIOIO(BaseTestIO, unittest.TestCase, ):
    ioclass = TdtIO
    files_to_test = ['aep_05']
    files_to_download = ['aep_05/Block-1/aep_05_Block-1.Tbk',
                         'aep_05/Block-1/aep_05_Block-1.Tdx',
                         'aep_05/Block-1/aep_05_Block-1.tev',
                         'aep_05/Block-1/aep_05_Block-1.tsq',

                         #~ 'aep_05/Block-2/aep_05_Block-2.Tbk',
                         #~ 'aep_05/Block-2/aep_05_Block-2.Tdx',
                         #~ 'aep_05/Block-2/aep_05_Block-2.tev',
                         #~ 'aep_05/Block-2/aep_05_Block-2.tsq',

                         #~ 'aep_05/Block-3/aep_05_Block-3.Tbk',
                         #~ 'aep_05/Block-3/aep_05_Block-3.Tdx',
                         #~ 'aep_05/Block-3/aep_05_Block-3.tev',
                         #~ 'aep_05/Block-3/aep_05_Block-3.tsq',
                         ]


if __name__ == "__main__":
    unittest.main()
