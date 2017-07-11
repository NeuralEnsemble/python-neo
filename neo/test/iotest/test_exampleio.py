# -*- coding: utf-8 -*-
"""
Tests of neo.io.exampleio
"""

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

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
        for anasig in seg.analogsignals:
            self.assertNotEqual(anasig.size, 0)
        for st in seg.spiketrains:
            self.assertNotEqual(st.size, 0)
        
        #annotations
        assert 'seg_extra_info' in seg.annotations
        assert seg.name=='Seg #0 Block #0'
        for anasig in seg.analogsignals:
            assert anasig.name is not None
        for st in seg.spiketrains:
            assert st.name is not None
        for ev in seg.events:
            assert ev.name is not None
        for ep in seg.epochs:
            assert ep.name is not None
    
    def test_read_block(self):
        r = ExampleIO(filename=None)
        bl = r.read_block(cascade=True, lazy=True)
        assert len(bl.list_units) == 3
        assert len(bl.channel_indexes) == 16 + 3 #signals + units



if __name__ == "__main__":
    unittest.main()
