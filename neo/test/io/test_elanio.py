# encoding: utf-8
"""
Tests of io.elanio
"""

from __future__ import division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io import ElanIO
import numpy


from neo.test.io.common_io_test import BaseTestIO





class TestElanIO(BaseTestIO , unittest.TestCase, ):
    ioclass = ElanIO
    files_to_test = [   'File_elan_1.eeg',
                            ]
    files_to_download =  [   'File_elan_1.eeg',
                                    'File_elan_1.eeg.ent',
                                    'File_elan_1.eeg.pos',
                                    ]



if __name__ == "__main__":
    unittest.main()
