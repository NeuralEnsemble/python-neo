# -*- coding: utf-8 -*-
"""
Tests of neo.io.exampleio
"""

# needed for python 3 compatibility
from __future__ import unicode_literals, print_function, division, absolute_import

import unittest

from neo.io.exampleio import ExampleIO#, HAVE_SCIPY
from neo.test.iotest.common_io_test import BaseTestIO

import quantities as pq
import numpy as np

#This run standart tests, this is mandatory for all IO
class TestExampleIO(BaseTestIO, unittest.TestCase, ):
    ioclass = ExampleIO
    files_to_test = ['fake1',
                     'fake2',
                     ]
    files_to_download = []


class Specific_TestExampleIO(unittest.TestCase):
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
        #~ print(len(bl.channel_indexes))
        assert len(bl.channel_indexes) == 1 + 3 #signals grouped + units

    def test_read_segment_with_time_slice(self):
        r = ExampleIO(filename=None)
        seg = r.read_segment(time_slice=None)
        shape_full = seg.analogsignals[0].shape
        spikes_full = seg.spiketrains[0]
        event_full = seg.events[0]
        
        t_start, t_stop = 260*pq.ms, 1.854*pq.s
        seg = r.read_segment(time_slice=(t_start, t_stop))
        shape_slice = seg.analogsignals[0].shape
        spikes_slice = seg.spiketrains[0]
        event_slice = seg.events[0]
        
        assert shape_full[0]>shape_slice[0]
        
        assert spikes_full.size>spikes_slice.size
        assert np.all(spikes_slice>=t_start)
        assert np.all(spikes_slice<=t_stop)
        assert spikes_slice.t_start==t_start
        assert spikes_slice.t_stop==t_stop
        
        assert event_full.size>event_slice.size
        assert np.all(event_slice.times>=t_start)
        assert np.all(event_slice.times<=t_stop)

    def test_read_block_with_time_slices(self):
        r = ExampleIO(filename=None)
        bl = r.read_block(time_slices=None)
        real_segments = bl.segments
        assert len(real_segments)==2
        
        
        time_slices = [(1, 3), (4, 5), (16, 21), (21.5, 22.)]
        bl = r.read_block(time_slices=time_slices)
        sliced_segments = bl.segments
        assert len(sliced_segments)==len(time_slices)
        
        with self.assertRaises(ValueError):
            buggy_time_slices = [(11, 14)]
            bl = r.read_block(time_slices=buggy_time_slices)
        
        
        
        
        



if __name__ == "__main__":
    unittest.main()


