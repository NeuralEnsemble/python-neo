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

from neo.io.exampleio import ExampleIO, HAVE_SCIPY
from neo.test.iotest.common_io_test import BaseTestIO


class TestExampleIO(BaseTestIO, unittest.TestCase, ):
    ioclass = ExampleIO
    files_to_test = ['fake1',
                     'fake2',
                     ]
    files_to_download = []


class TestExample2IO(unittest.TestCase):
    @unittest.skipUnless(HAVE_SCIPY, "requires scipy")
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

    @unittest.skipUnless(HAVE_SCIPY, "requires scipy")
    def test_read_segment_cascade(self):
        r = ExampleIO(filename=None)
        seg = r.read_segment(cascade=False)
        self.assertEqual(len(seg.analogsignals), 0)
        seg = r.read_segment(cascade=True, num_analogsignal=4)
        self.assertEqual(len(seg.analogsignals), 4)

    @unittest.skipUnless(HAVE_SCIPY, "requires scipy")
    def test_read_analogsignal(self):
        r = ExampleIO(filename=None)
        r.read_analogsignal(lazy=False, segment_duration=15., t_start=-1)

    @unittest.skipUnless(HAVE_SCIPY, "requires scipy")
    def read_spiketrain(self):
        r = ExampleIO(filename=None)
        r.read_spiketrain(lazy=False,)


if __name__ == "__main__":
    unittest.main()
