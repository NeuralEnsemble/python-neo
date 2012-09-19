# encoding: utf-8
"""
Tests of io.elanio
"""
from __future__ import absolute_import, division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from ...io import BrainVisionIO
import numpy


from .common_io_test import BaseTestIO





class TestBrainVisionIO(BaseTestIO , unittest.TestCase, ):
    ioclass = BrainVisionIO
    files_to_test = [   'File_brainvision_1.vhdr',
                                'File_brainvision_2.vhdr',
                            ]
    files_to_download =  [  'File_brainvision_1.eeg',
                                        'File_brainvision_1.vhdr',
                                        'File_brainvision_1.vmrk',
                                        
                                        'File_brainvision_2.eeg',
                                        'File_brainvision_2.vhdr',
                                        'File_brainvision_2.vmrk',
                                    ]


if __name__ == "__main__":
    unittest.main()
