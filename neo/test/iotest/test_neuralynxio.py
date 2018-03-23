# -*- coding: utf-8 -*-
"""
Tests of neo.io.blackrockio
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import time
import warnings

import unittest

import numpy as np
import quantities as pq

from neo.test.iotest.common_io_test import BaseTestIO
from neo.core import *

from neo.io.neuralynxio import NeuralynxIO
from neo.io.neuralynxio import NeuralynxIO as NewNeuralynxIO
from neo.io.neuralynxio_v1 import NeuralynxIO as OldNeuralynxIO
from neo import AnalogSignal


class CommonNeuralynxIOTest(BaseTestIO, unittest.TestCase, ):
    ioclass = NeuralynxIO
    files_to_test = [
        'Cheetah_v5.5.1/original_data',
        'Cheetah_v5.6.3/original_data',
        'Cheetah_v5.7.4/original_data',
    ]
    files_to_download = [
        'Cheetah_v5.5.1/original_data/CheetahLogFile.txt',
        'Cheetah_v5.5.1/original_data/CheetahLostADRecords.txt',
        'Cheetah_v5.5.1/original_data/Events.nev',
        'Cheetah_v5.5.1/original_data/STet3a.nse',
        'Cheetah_v5.5.1/original_data/STet3b.nse',
        'Cheetah_v5.5.1/original_data/Tet3a.ncs',
        'Cheetah_v5.5.1/original_data/Tet3b.ncs',
        'Cheetah_v5.5.1/plain_data/STet3a.txt',
        'Cheetah_v5.5.1/plain_data/STet3b.txt',
        'Cheetah_v5.5.1/plain_data/Tet3a.txt',
        'Cheetah_v5.5.1/plain_data/Tet3b.txt',
        'Cheetah_v5.5.1/plain_data/Events.txt',
        'Cheetah_v5.5.1/README.txt',
        'Cheetah_v5.6.3/original_data/CheetahLogFile.txt',
        'Cheetah_v5.6.3/original_data/CheetahLostADRecords.txt',
        'Cheetah_v5.6.3/original_data/Events.nev',
        'Cheetah_v5.6.3/original_data/CSC1.ncs',
        'Cheetah_v5.6.3/original_data/CSC2.ncs',
        'Cheetah_v5.6.3/original_data/TT1.ntt',
        'Cheetah_v5.6.3/original_data/TT2.ntt',
        'Cheetah_v5.6.3/original_data/VT1.nvt',
        'Cheetah_v5.6.3/plain_data/Events.txt',
        'Cheetah_v5.6.3/plain_data/CSC1.txt',
        'Cheetah_v5.6.3/plain_data/CSC2.txt',
        'Cheetah_v5.6.3/plain_data/TT1.txt',
        'Cheetah_v5.6.3/plain_data/TT2.txt',
        'Cheetah_v5.6.3/original_data/VT1.nvt',
        'Cheetah_v5.7.4/original_data/CSC1.ncs',
        'Cheetah_v5.7.4/original_data/CSC2.ncs',
        'Cheetah_v5.7.4/original_data/CSC3.ncs',
        'Cheetah_v5.7.4/original_data/CSC4.ncs',
        'Cheetah_v5.7.4/original_data/CSC5.ncs',
        'Cheetah_v5.7.4/original_data/Events.nev',
        'Cheetah_v5.7.4/plain_data/CSC1.txt',
        'Cheetah_v5.7.4/plain_data/CSC2.txt',
        'Cheetah_v5.7.4/plain_data/CSC3.txt',
        'Cheetah_v5.7.4/plain_data/CSC4.txt',
        'Cheetah_v5.7.4/plain_data/CSC5.txt',
        'Cheetah_v5.7.4/plain_data/Events.txt',
        'Cheetah_v5.7.4/README.txt']


class TestCheetah_v551(CommonNeuralynxIOTest, unittest.TestCase):
    cheetah_version = '5.5.1'
    files_to_test = []

    def test_read_block(self):
        """Read data in a certain time range into one block"""
        dirname = self.get_filename_path('Cheetah_v5.5.1/original_data')
        nio = NeuralynxIO(dirname=dirname, use_cache=False)

        block = nio.read_block()

        # Everything put in one segment
        self.assertEqual(len(block.segments), 2)
        seg = block.segments[0]
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(seg.analogsignals[0].shape[-1], 2)

        self.assertEqual(seg.analogsignals[0].sampling_rate, 32. * pq.kHz)
        self.assertEqual(len(seg.spiketrains), 2)

        # Testing different parameter combinations
        block = nio.read_block(lazy=True)
        self.assertEqual(len(block.segments[0].analogsignals[0]), 0)
        self.assertEqual(len(block.segments[0].spiketrains[0]), 0)

        block = nio.read_block(load_waveforms=True)
        self.assertEqual(len(block.segments[0].analogsignals), 1)
        self.assertEqual(len(block.segments[0].spiketrains), 2)
        self.assertEqual(block.segments[0].spiketrains[0].waveforms.shape[0],
                         block.segments[0].spiketrains[0].shape[0])
        self.assertGreater(len(block.segments[0].events), 0)

        self.assertEqual(len(block.channel_indexes[-1].units[0].spiketrains), 2)  # 2 segment

        block = nio.read_block(load_waveforms=True, units_group_mode='all-in-one')
        self.assertEqual(len(block.channel_indexes[-1].units), 2)  # 2 units

        block = nio.read_block(load_waveforms=True, units_group_mode='split-all')
        self.assertEqual(len(block.channel_indexes[-1].units), 1)  # 1 units by ChannelIndex

    def test_read_segment(self):
        dirname = self.get_filename_path('Cheetah_v5.5.1/original_data')
        nio = NeuralynxIO(dirname=dirname, use_cache=False)

        # read first segment entirely
        seg = nio.read_segment(seg_index=0, time_slice=None)
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(seg.analogsignals[0].shape[-1], 2)
        self.assertEqual(seg.analogsignals[0].sampling_rate, 32 * pq.kHz)
        self.assertEqual(len(seg.spiketrains), 2)

        # Testing different parameter combinations
        seg = nio.read_segment(seg_index=0, lazy=True)
        self.assertEqual(seg.analogsignals[0].size, 0)
        self.assertEqual(seg.spiketrains[0].size, 0)

        seg = nio.read_segment(seg_index=0, load_waveforms=True)
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(len(seg.spiketrains), 2)
        self.assertTrue(len(seg.spiketrains[0].waveforms) > 0)
        self.assertTrue(len(seg.events) > 0)


class TestCheetah_v563(CommonNeuralynxIOTest, unittest.TestCase):
    cheetah_version = '5.6.3'
    files_to_test = []

    def test_read_block(self):
        """Read data in a certain time range into one block"""
        dirname = self.get_filename_path('Cheetah_v5.6.3/original_data')
        nio = NeuralynxIO(dirname=dirname, use_cache=False)

        block = nio.read_block()

        # There are two segments due to gap in recording
        self.assertEqual(len(block.segments), 2)
        for seg in block.segments:
            self.assertEqual(len(seg.analogsignals), 1)
            self.assertEqual(seg.analogsignals[0].shape[-1], 2)
            self.assertEqual(seg.analogsignals[0].sampling_rate, 2. * pq.kHz)
            self.assertEqual(len(seg.spiketrains), 8)

        # Testing different parameter combinations
        block = nio.read_block(lazy=True)
        self.assertEqual(len(block.segments[0].analogsignals[0]), 0)
        self.assertEqual(len(block.segments[0].spiketrains[0]), 0)

        block = nio.read_block(load_waveforms=True)
        self.assertEqual(len(block.segments[0].analogsignals), 1)
        self.assertEqual(len(block.segments[0].spiketrains), 8)
        self.assertEqual(block.segments[0].spiketrains[0].waveforms.shape[0],
                         block.segments[0].spiketrains[0].shape[0])
        # this is tetrode data, containing 32 samples per waveform
        self.assertEqual(block.segments[0].spiketrains[0].waveforms.shape[1], 4)
        self.assertEqual(block.segments[0].spiketrains[0].waveforms.shape[-1], 32)
        self.assertGreater(len(block.segments[0].events), 0)

        self.assertEqual(len(block.channel_indexes[-1].units[0].spiketrains), 2)

        block = nio.read_block(load_waveforms=True, units_group_mode='all-in-one')
        self.assertEqual(len(block.channel_indexes[-1].units), 8)

        block = nio.read_block(load_waveforms=True, units_group_mode='split-all')
        self.assertEqual(len(block.channel_indexes[-1].units), 1)  # 1 units by ChannelIndex

    def test_read_segment(self):
        dirname = self.get_filename_path('Cheetah_v5.5.1/original_data')
        nio = NeuralynxIO(dirname=dirname, use_cache=False)

        # read first segment entirely
        seg = nio.read_segment(seg_index=0, time_slice=None)
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(seg.analogsignals[0].shape[-1], 2)
        self.assertEqual(seg.analogsignals[0].sampling_rate, 32 * pq.kHz)
        self.assertEqual(len(seg.spiketrains), 2)

        # Testing different parameter combinations
        seg = nio.read_segment(seg_index=0, lazy=True)
        self.assertEqual(seg.analogsignals[0].size, 0)
        self.assertEqual(seg.spiketrains[0].size, 0)

        seg = nio.read_segment(seg_index=0, load_waveforms=True)
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(len(seg.spiketrains), 2)
        self.assertTrue(len(seg.spiketrains[0].waveforms) > 0)
        self.assertTrue(len(seg.events) > 0)


class TestCheetah_v574(CommonNeuralynxIOTest, unittest.TestCase):
    cheetah_version = '5.7.4'
    files_to_test = []

    def test_read_block(self):
        dirname = self.get_filename_path('Cheetah_v5.7.4/original_data')
        nio = NeuralynxIO(dirname=dirname, use_cache=False)

        block = nio.read_block()

        # Everything put in one segment
        seg = block.segments[0]
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(seg.analogsignals[0].shape[-1], 5)

        self.assertEqual(seg.analogsignals[0].sampling_rate, 32 * pq.kHz)
        self.assertEqual(len(seg.spiketrains), 0)  # no nse files available

        # Testing different parameter combinations
        block = nio.read_block(lazy=True)
        self.assertEqual(len(block.segments[0].analogsignals[0]), 0)

        block = nio.read_block(load_waveforms=True)
        self.assertEqual(len(block.segments[0].analogsignals), 1)
        self.assertEqual(len(block.segments[0].spiketrains), 0)
        self.assertGreater(len(block.segments[0].events), 0)

        block = nio.read_block(signal_group_mode='split-all')
        self.assertEqual(len(block.channel_indexes), 5)

        block = nio.read_block(signal_group_mode='group-by-same-units')
        self.assertEqual(len(block.channel_indexes), 1)


class TestData(CommonNeuralynxIOTest, unittest.TestCase):
    def test_ncs(self):
        for session in self.files_to_test[1:2]:  # in the long run this should include all files
            dirname = self.get_filename_path(session)
            nio = NeuralynxIO(dirname=dirname, use_cache=False)
            block = nio.read_block()

            for anasig_id, anasig in enumerate(block.segments[0].analogsignals):
                chid = anasig.channel_index.annotations['channel_id'][anasig_id]
                filename = nio.ncs_filenames[chid][:-3] + 'txt'
                filename = filename.replace('original_data', 'plain_data')
                plain_data = np.loadtxt(filename)[:, 5:].flatten()  # first columns are meta info
                overlap = 512 * 500
                gain_factor_0 = plain_data[0] / anasig.magnitude[0, 0]
                np.testing.assert_allclose(plain_data[:overlap],
                                           anasig.magnitude[:overlap, 0] * gain_factor_0,
                                           rtol=0.01)


class TestGaps(CommonNeuralynxIOTest, unittest.TestCase):
    def test_gap_handling_v551(self):
        dirname = self.get_filename_path('Cheetah_v5.5.1/original_data')
        nio = NeuralynxIO(dirname=dirname, use_cache=False)

        block = nio.read_block()

        # known gap values
        n_gaps = 1
        # so 2 segments, 2 anasigs by Channelindex, 2 SpikeTrain by Units
        self.assertEqual(len(block.segments), n_gaps + 1)
        self.assertEqual(len(block.channel_indexes[0].analogsignals), n_gaps + 1)
        self.assertEqual(len(block.channel_indexes[-1].units[0].spiketrains), n_gaps + 1)

    def test_gap_handling_v563(self):
        dirname = self.get_filename_path('Cheetah_v5.6.3/original_data')
        nio = NeuralynxIO(dirname=dirname, use_cache=False)
        block = nio.read_block()

        # known gap values
        n_gaps = 1
        # so 2 segments, 2 anasigs by Channelindex, 2 SpikeTrain by Units
        self.assertEqual(len(block.segments), n_gaps + 1)
        self.assertEqual(len(block.channel_indexes[0].analogsignals), n_gaps + 1)
        self.assertEqual(len(block.channel_indexes[-1].units[0].spiketrains), n_gaps + 1)


def compare_old_and_new_neuralynxio():
    base = '/tmp/files_for_testing_neo/neuralynx/'
    dirname = base + 'Cheetah_v5.5.1/original_data/'
    # ~ dirname = base+'Cheetah_v5.7.4/original_data/'

    t0 = time.perf_counter()
    newreader = NewNeuralynxIO(dirname)
    t1 = time.perf_counter()
    bl1 = newreader.read_block(load_waveforms=True)
    t2 = time.perf_counter()
    print('newreader header', t1 - t0, 's')
    print('newreader data', t2 - t1, 's')
    print('newreader toal', t2 - t0, 's')
    for seg in bl1.segments:
        print('seg', seg.index)
        for anasig in seg.analogsignals:
            print(' AnalogSignal', anasig.name, anasig.shape, anasig.t_start)
        for st in seg.spiketrains:
            print(' SpikeTrain', st.name, st.shape, st.waveforms.shape, st[:5])
        for ev in seg.events:
            print(' Event', ev.name, ev.times.shape)

    print('*' * 10)

    t0 = time.perf_counter()
    oldreader = OldNeuralynxIO(sessiondir=dirname, use_cache='never')
    t1 = time.perf_counter()
    bl2 = oldreader.read_block(waveforms=True, events=True)
    t2 = time.perf_counter()
    print('oldreader header', t1 - t0, 's')
    print('oldreader data', t2 - t1, 's')
    print('oldreader toal', t2 - t0, 's')
    for seg in bl2.segments:
        print('seg', seg.index)
        for anasig in seg.analogsignals:
            print(' AnalogSignal', anasig.name, anasig.shape, anasig.t_start)
        for st in seg.spiketrains:
            print(' SpikeTrain', st.name, st.shape, st.waveforms.shape, st[:5])
        for ev in seg.events:
            print(' Event', ev.name, ev.times.shape)

    print('*' * 10)
    compare_neo_content(bl1, bl2)


def compare_neo_content(bl1, bl2):
    print('*' * 5, 'Comparison of blocks', '*' * 5)
    object_types_to_test = [Segment, ChannelIndex, Unit, AnalogSignal,
                            SpikeTrain, Event, Epoch]
    for objtype in object_types_to_test:
        print('Testing {}'.format(objtype))
        children1 = bl1.list_children_by_class(objtype)
        children2 = bl2.list_children_by_class(objtype)

        if len(children1) != len(children2):
            warnings.warn('Number of {} is different in both blocks ({} != {}).'
                          ' Skipping comparison'
                          ''.format(objtype, len(children1), len(children2)))
            continue

        for child1, child2 in zip(children1, children2):
            compare_annotations(child1.annotations, child2.annotations)
            compare_attributes(child1, child2)


def compare_annotations(anno1, anno2):
    if len(anno1) != len(anno2):
        warnings.warn('Different numbers of annotations! {} != {}\nSkipping further comparison of '
                      'this annotation list.'.format(anno1.keys(), anno2.keys()))
        return
    assert anno1.keys() == anno2.keys()
    for key in anno1.keys():
        anno1[key] = anno2[key]


def compare_attributes(child1, child2):
    assert child1._all_attrs == child2._all_attrs
    for attr_id in range(len(child1._all_attrs)):
        attr_name = child1._all_attrs[attr_id][0]
        attr_dtype = child1._all_attrs[attr_id][1]
        if type(child1) == AnalogSignal and attr_name == 'signal':
            continue
        if type(child1) == SpikeTrain and attr_name == 'times':
            continue
        unequal = child1.__getattribute__(attr_name) != child2.__getattribute__(attr_name)

        if hasattr(unequal, 'any'):
            unequal = unequal.any()
        if unequal:
            warnings.warn('Attributes differ! {}.{}={} is not equal to {}.{}={}'
                          ''.format(child1.__class__.__name__, attr_name,
                                    child1.__getattribute__(attr_name),
                                    child2.__class__.__name__, attr_name,
                                    child2.__getattribute__(attr_name)))


if __name__ == '__main__':
    unittest.main()
    # ~ compare_old_and_new_neuralynxio()
