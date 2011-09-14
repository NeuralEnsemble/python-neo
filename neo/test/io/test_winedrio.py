# encoding: utf-8
"""
Tests of io.wineedrio
"""

from __future__ import division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import WinEdrIO
import numpy


from neo.test.io.common_io_test import BaseTestIO



class TestWinedrIO(BaseTestIO , unittest.TestCase, ):
    ioclass = WinEdrIO
    files_to_test = [   'File_WinEDR_1.EDR',
                                'File_WinEDR_2.EDR',
                                'File_WinEDR_3.EDR',
                            ]
    files_to_download = files_to_test



if __name__ == "__main__":
    unittest.main()
