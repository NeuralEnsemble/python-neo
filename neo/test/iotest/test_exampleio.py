# -*- coding: utf-8 -*-
"""
Tests of neo.io.exampleio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io.exampleio import ExampleIO#, HAVE_SCIPY
from neo.test.iotest.common_io_test import BaseTestIO


class TestExampleIO(BaseTestIO, unittest.TestCase, ):
    ioclass = ExampleIO
    files_to_test = ['fake1',
                     'fake2',
                     ]
    files_to_download = []


class TestExample2IO(unittest.TestCase):

    def test_read_segment_lazy(self):
        r = ExampleIO(filename=None)
        seg = r.read_segment(cascade=True, lazy=True)
        for ana in seg.analogsignals:
            self.assertEqual(ana.size, 0)
            assert hasattr(ana, 'lazy_shape')
        for st in seg.spiketrains:
            self.assertEqual(st.size, 0)
            assert hasattr(st, 'lazy_shape')

        seg = r.read_segment(cascade=True, lazy=False)
        for ana in seg.analogsignals:
            self.assertNotEqual(ana.size, 0)
        for st in seg.spiketrains:
            self.assertNotEqual(st.size, 0)
        
        assert 'seg_extra_info' in seg.annotations
        assert seg.name=='Seg #0 Block #0'
        
        r.print_annotations()


if __name__ == "__main__":
    unittest.main()
