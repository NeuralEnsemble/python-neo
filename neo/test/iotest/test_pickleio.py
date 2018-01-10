# -*- coding: utf-8 -*-
"""
Tests of the neo.io.pickleio.PickleIO class
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

import os

import unittest

import numpy as np
import quantities as pq

from neo.core import Block, Segment, AnalogSignal, SpikeTrain, Unit, Epoch, Event, ChannelIndex, IrregularlySampledSignal
from neo.io import PickleIO
from numpy.testing import assert_array_equal
from neo.test.tools import assert_arrays_equal, assert_file_contents_equal
from neo.test.iotest.common_io_test import BaseTestIO



NCELLS = 5


class CommonTestPickleIO(BaseTestIO, unittest.TestCase):
    ioclass = PickleIO

    def test_readed_with_cascade_is_compliant(self):
        pass
    test_readed_with_cascade_is_compliant.__test__ = False  # PickleIO does not support lazy loading

    def test_readed_with_lazy_is_compliant(self):
        pass
    test_readed_with_lazy_is_compliant.__test__ = False


class TestPickleIO(unittest.TestCase):

    def test__issue_285(self):
        ##Spiketrain
        train = SpikeTrain([3, 4, 5] * pq.s, t_stop=10.0)
        unit = Unit()
        train.unit = unit
        unit.spiketrains.append(train)

        epoch = Epoch([0, 10, 20], [2, 2, 2], ["a", "b", "c"], units="ms")

        blk = Block()
        seg = Segment()
        seg.spiketrains.append(train)
        seg.epochs.append(epoch)
        epoch.segment = seg
        blk.segments.append(seg)

        reader = PickleIO(filename="blk.pkl")
        reader.write(blk)

        reader = PickleIO(filename="blk.pkl")
        r_blk = reader.read_block()
        r_seg = r_blk.segments[0]
        self.assertIsInstance(r_seg.spiketrains[0].unit, Unit)
        self.assertIsInstance(r_seg.epochs[0], Epoch)
        os.remove('blk.pkl')

        ##Epoch
        epoch = Epoch(times=np.arange(0, 30, 10)*pq.s,durations=[10, 5, 7]*pq.ms,labels=np.array(['btn0', 'btn1', 'btn2'], dtype='S'))
        epoch.segment = Segment()
        blk = Block()
        seg = Segment()
        seg.epochs.append(epoch)
        blk.segments.append(seg)

        reader = PickleIO(filename="blk.pkl")
        reader.write(blk)

        reader = PickleIO(filename="blk.pkl")
        r_blk = reader.read_block()
        r_seg = r_blk.segments[0]
        self.assertIsInstance(r_seg.epochs[0].segment, Segment)
        os.remove('blk.pkl')

        ##Event
        event = Event(np.arange(0, 30, 10)*pq.s,labels=np.array(['trig0', 'trig1', 'trig2'],dtype='S'))
        event.segment = Segment()

        blk = Block()
        seg = Segment()
        seg.events.append(event)
        blk.segments.append(seg)

        reader = PickleIO(filename="blk.pkl")
        reader.write(blk)

        reader = PickleIO(filename="blk.pkl")
        r_blk = reader.read_block()
        r_seg = r_blk.segments[0]
        self.assertIsInstance(r_seg.events[0].segment, Segment)
        os.remove('blk.pkl')

        ##IrregularlySampledSignal
        signal =  IrregularlySampledSignal([0.0, 1.23, 6.78], [1, 2, 3],units='mV', time_units='ms')
        signal.segment = Segment()

        blk = Block()
        seg = Segment()
        seg.irregularlysampledsignals.append(signal)
        blk.segments.append(seg)
        blk.segments[0].block = blk

        reader = PickleIO(filename="blk.pkl")
        reader.write(blk)

        reader = PickleIO(filename="blk.pkl")
        r_blk = reader.read_block()
        r_seg = r_blk.segments[0]
        self.assertIsInstance(r_seg.irregularlysampledsignals[0].segment, Segment)
        os.remove('blk.pkl')

if __name__ == '__main__':
    unittest.main()
