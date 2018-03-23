# -*- coding: utf-8 -*-
"""
Tests of neo.io.NSDFIO
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

import numpy as np
import quantities as pq
from datetime import datetime
import os

import unittest

from neo.io.nsdfio import HAVE_NSDF, NSDFIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.core import AnalogSignal, Segment, Block, ChannelIndex
from neo.test.tools import assert_same_attributes, assert_same_annotations, \
    assert_neo_object_is_compliant


@unittest.skipUnless(HAVE_NSDF, "Requires NSDF")
class CommonTests(BaseTestIO, unittest.TestCase):
    ioclass = NSDFIO
    read_and_write_is_bijective = False


@unittest.skipUnless(HAVE_NSDF, "Requires NSDF")
class NSDFIOTest(unittest.TestCase):
    """
    Base class for all NSDFIO tests.

    setUp and tearDown methods are responsible for respectively: setting up and cleaning after tests
    All create_{object} methods create and return an example {object}.
    """

    def setUp(self):
        self.filename = 'nsdfio_testfile.h5'
        self.io = NSDFIO(self.filename)

    def tearDown(self):
        os.remove(self.filename)

    def create_list_of_blocks(self):
        blocks = []

        for i in range(2):
            blocks.append(self.create_block(name='Block #{}'.format(i)))

        return blocks

    def create_block(self, name='Block'):
        block = Block()

        self._assign_basic_attributes(block, name=name)
        self._assign_datetime_attributes(block)
        self._assign_index_attribute(block)

        self._create_block_children(block)

        self._assign_annotations(block)

        return block

    def _create_block_children(self, block):
        for i in range(3):
            block.segments.append(self.create_segment(block, name='Segment #{}'.format(i)))
        for i in range(3):
            block.channel_indexes.append(
                self.create_channelindex(block, name='ChannelIndex #{}'.format(i),
                                         analogsignals=[seg.analogsignals[i] for seg in
                                                        block.segments]))

    def create_segment(self, parent=None, name='Segment'):
        segment = Segment()

        segment.block = parent

        self._assign_basic_attributes(segment, name=name)
        self._assign_datetime_attributes(segment)
        self._assign_index_attribute(segment)

        self._create_segment_children(segment)

        self._assign_annotations(segment)

        return segment

    def _create_segment_children(self, segment):
        for i in range(2):
            segment.analogsignals.append(self.create_analogsignal(
                segment, name='Signal #{}'.format(i * 3)))
            segment.analogsignals.append(self.create_analogsignal2(
                segment, name='Signal #{}'.format(i * 3 + 1)))
            segment.analogsignals.append(self.create_analogsignal3(
                segment, name='Signal #{}'.format(i * 3 + 2)))

    def create_analogsignal(self, parent=None, name='AnalogSignal1'):
        signal = AnalogSignal([[1.0, 2.5], [2.2, 3.1], [3.2, 4.4]], units='mV',
                              sampling_rate=100 * pq.Hz, t_start=2 * pq.min)

        signal.segment = parent
        self._assign_basic_attributes(signal, name=name)
        self._assign_annotations(signal)

        return signal

    def create_analogsignal2(self, parent=None, name='AnalogSignal2'):
        signal = AnalogSignal([[1], [2], [3], [4], [5]], units='mA',
                              sampling_period=0.5 * pq.ms)

        signal.segment = parent
        self._assign_annotations(signal)

        return signal

    def create_analogsignal3(self, parent=None, name='AnalogSignal3'):
        signal = AnalogSignal([[1, 2, 3], [4, 5, 6]], units='mV',
                              sampling_rate=2 * pq.kHz, t_start=100 * pq.s)

        signal.segment = parent
        self._assign_basic_attributes(signal, name=name)

        return signal

    def create_channelindex(self, parent=None, name='ChannelIndex', analogsignals=None):
        channels_num = min([signal.shape[1] for signal in analogsignals])

        channelindex = ChannelIndex(index=np.arange(channels_num),
                                    channel_names=['Channel{}'.format(
                                        i) for i in range(channels_num)],
                                    channel_ids=np.arange(channels_num),
                                    coordinates=([[1.87, -5.2, 4.0]] * channels_num) * pq.cm)

        for signal in analogsignals:
            channelindex.analogsignals.append(signal)

        self._assign_basic_attributes(channelindex, name)
        self._assign_annotations(channelindex)

        return channelindex

    def _assign_basic_attributes(self, object, name=None):
        if name is None:
            object.name = 'neo object'
        else:
            object.name = name
        object.description = 'Example of neo object'
        object.file_origin = 'datafile.pp'

    def _assign_datetime_attributes(self, object):
        object.file_datetime = datetime(2017, 6, 11, 14, 53, 23)
        object.rec_datetime = datetime(2017, 5, 29, 13, 12, 47)

    def _assign_index_attribute(self, object):
        object.index = 12

    def _assign_annotations(self, object):
        object.annotations = {'str': 'value',
                              'int': 56,
                              'float': 5.234}


@unittest.skipUnless(HAVE_NSDF, "Requires NSDF")
class NSDFIOTestWriteThenRead(NSDFIOTest):
    """
    Class for testing NSDFIO.
    It first creates example neo objects, then writes them to the file,
    reads the file and compares the result with the original ones.

    all test_{object} methods run "write then read" test for a/an {object}
    all compare_{object} methods check if the second {object} is a proper copy
        of the first one
    """
    lazy_modes = [False]

    def test_list_of_blocks(self, lazy=False):
        blocks = self.create_list_of_blocks()
        self.io.write(blocks)
        for lazy in self.lazy_modes:
            blocks2 = self.io.read(lazy=lazy)
            self.compare_list_of_blocks(blocks, blocks2, lazy)

    def test_block(self, lazy=False):
        block = self.create_block()
        self.io.write_block(block)
        for lazy in self.lazy_modes:
            block2 = self.io.read_block(lazy=lazy)
            self.compare_blocks(block, block2, lazy)

    def test_segment(self, lazy=False):
        segment = self.create_segment()
        self.io.write_segment(segment)
        for lazy in self.lazy_modes:
            segment2 = self.io.read_segment(lazy=lazy)
            self.compare_segments(segment, segment2, lazy)

    def compare_list_of_blocks(self, blocks1, blocks2, lazy=False):
        assert len(blocks1) == len(blocks2)
        for block1, block2 in zip(blocks1, blocks2):
            self.compare_blocks(block1, block2, lazy)

    def compare_blocks(self, block1, block2, lazy=False):
        self._compare_objects(block1, block2)
        assert block2.file_datetime == datetime.fromtimestamp(os.stat(self.filename).st_mtime)
        assert_neo_object_is_compliant(block2)
        self._compare_blocks_children(block1, block2, lazy=lazy)

    def _compare_blocks_children(self, block1, block2, lazy):
        assert len(block1.segments) == len(block2.segments)
        for segment1, segment2 in zip(block1.segments, block2.segments):
            self.compare_segments(segment1, segment2, lazy=lazy)

    def compare_segments(self, segment1, segment2, lazy=False):
        self._compare_objects(segment1, segment2)
        assert segment2.file_datetime == datetime.fromtimestamp(os.stat(self.filename).st_mtime)
        self._compare_segments_children(segment1, segment2, lazy=lazy)

    def _compare_segments_children(self, segment1, segment2, lazy):
        assert len(segment1.analogsignals) == len(segment2.analogsignals)
        for signal1, signal2 in zip(segment1.analogsignals, segment2.analogsignals):
            self.compare_analogsignals(signal1, signal2, lazy=lazy)

    def compare_analogsignals(self, signal1, signal2, lazy=False):
        if not lazy:
            self._compare_objects(signal1, signal2)
        else:
            self._compare_objects(signal1, signal2, exclude_attr=['shape', 'signal'])
            assert signal2.lazy_shape == signal1.shape
        assert signal2.dtype == signal1.dtype

    def _compare_objects(self, object1, object2, exclude_attr=[]):
        assert object1.__class__.__name__ == object2.__class__.__name__
        assert object2.file_origin == self.filename
        assert_same_attributes(object1, object2, exclude=[
                                                             'file_origin',
                                                             'file_datetime'] + exclude_attr)
        assert_same_annotations(object1, object2)


if __name__ == "__main__":
    unittest.main()
