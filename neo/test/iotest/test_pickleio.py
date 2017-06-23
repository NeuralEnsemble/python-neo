# -*- coding: utf-8 -*-
"""
Tests of the neo.io.pickleio.PickleIO class
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

import os

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

from neo.core import Block, Segment, AnalogSignal, SpikeTrain, Unit, Epoch, Event
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


if __name__ == '__main__':
    unittest.main()
