# -*- coding: utf-8 -*-
"""
Tests of neo.io.axonio
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import sys

import unittest

from neo.io import AxonIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestAxonIO(BaseTestIO, unittest.TestCase):
    files_to_test = ['File_axon_1.abf',
                     'File_axon_2.abf',
                     'File_axon_3.abf',
                     'File_axon_4.abf',
                     'File_axon_5.abf',
                     'File_axon_6.abf',
                     'File_axon_7.abf',
                     
                     ]
    files_to_download = files_to_test
    ioclass = AxonIO
    
    def test_annotations(self):
        reader = AxonIO(filename=self.get_filename_path('File_axon_2.abf'))
        bl = reader.read_block()
        ev = bl.segments[0].events[0]
        assert 'comments' in ev.annotations
    
    def test_read_protocol(self):
        for f in self.files_to_test:
            filename = self.get_filename_path(f)
            reader = AxonIO(filename=filename)
            bl = reader.read_block(lazy=True)
            if bl.annotations['abf_version']>=2.:
                reader.read_protocol()
        

if __name__ == "__main__":
    unittest.main()
