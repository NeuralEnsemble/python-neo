"""
Tests of neo.io.neuralynxio.py
"""

import os
import warnings

import unittest

import numpy as np
import quantities as pq

from neo.test.iotest.common_io_test import BaseTestIO
from neo.core import *

from neo.io.neuralynxio import NeuralynxIO
from neo import AnalogSignal

from neo.test.rawiotest.test_neuralynxrawio import TestNeuralynxRawIO

class CommonNeuralynxIOTest(BaseTestIO, unittest.TestCase, ):
    ioclass = NeuralynxIO
    entities_to_download = TestNeuralynxRawIO.entities_to_download
    entities_to_test = TestNeuralynxRawIO.entities_to_test


class TestCheetah_Neuraview(CommonNeuralynxIOTest, unittest.TestCase):
    files_to_test = []

    def test_read_block(self):
        dirname = self.get_local_path('neuralynx/Neuraview_v2/original_data')
        nio = NeuralynxIO(dirname=dirname, use_cache=False)
        bl = nio.read_block()

        # This dataset contains two event sets
        self.assertEqual(len(bl.segments[0].events), 2)


class TestCheetah_v551(CommonNeuralynxIOTest, unittest.TestCase):
    cheetah_version = '5.5.1'
    files_to_test = []

    def test_read_block(self):
        """Read data in a certain time range into one block"""
        dirname = self.get_local_path('neuralynx/Cheetah_v5.5.1/original_data')
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
        block = nio.read_block(load_waveforms=True)
        self.assertEqual(len(block.segments[0].analogsignals), 1)
        self.assertEqual(len(block.segments[0].spiketrains), 2)
        self.assertEqual(block.segments[0].spiketrains[0].waveforms.shape[0],
                         block.segments[0].spiketrains[0].shape[0])
        self.assertGreater(len(block.segments[0].events), 0)

        # self.assertEqual(len(block.channel_indexes[-1].units[0].spiketrains), 2)  # 2 segment

        # block = nio.read_block(load_waveforms=True, units_group_mode='all-in-one')
        # self.assertEqual(len(block.channel_indexes[-1].units), 2)  # 2 units

        # block = nio.read_block(load_waveforms=True, units_group_mode='split-all')
        # self.assertEqual(len(block.channel_indexes[-1].units), 1)  # 1 units by ChannelIndex

    def test_read_segment(self):
        dirname = self.get_local_path('neuralynx/Cheetah_v5.5.1/original_data')
        nio = NeuralynxIO(dirname=dirname, use_cache=False)

        # read first segment entirely
        seg = nio.read_segment(seg_index=0, time_slice=None)
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(seg.analogsignals[0].shape[-1], 2)
        self.assertEqual(seg.analogsignals[0].sampling_rate, 32 * pq.kHz)
        self.assertEqual(len(seg.spiketrains), 2)

        # Testing different parameter combinations
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
        dirname = self.get_local_path('neuralynx/Cheetah_v5.6.3/original_data')
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
        block = nio.read_block(load_waveforms=True)
        self.assertEqual(len(block.segments[0].analogsignals), 1)
        self.assertEqual(len(block.segments[0].spiketrains), 8)
        self.assertEqual(block.segments[0].spiketrains[0].waveforms.shape[0],
                         block.segments[0].spiketrains[0].shape[0])
        # this is tetrode data, containing 32 samples per waveform
        self.assertEqual(block.segments[0].spiketrains[0].waveforms.shape[1], 4)
        self.assertEqual(block.segments[0].spiketrains[0].waveforms.shape[-1], 32)
        self.assertGreater(len(block.segments[0].events), 0)

        # self.assertEqual(len(block.channel_indexes[-1].units[0].spiketrains), 2)

        # block = nio.read_block(load_waveforms=True, units_group_mode='all-in-one')
        # self.assertEqual(len(block.channel_indexes[-1].units), 8)

        # block = nio.read_block(load_waveforms=True, units_group_mode='split-all')
        # self.assertEqual(len(block.channel_indexes[-1].units), 1)  # 1 units by ChannelIndex

    def test_read_segment(self):
        dirname = self.get_local_path('neuralynx/Cheetah_v5.5.1/original_data')
        nio = NeuralynxIO(dirname=dirname, use_cache=False)

        # read first segment entirely
        seg = nio.read_segment(seg_index=0, time_slice=None)
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(seg.analogsignals[0].shape[-1], 2)
        self.assertEqual(seg.analogsignals[0].sampling_rate, 32 * pq.kHz)
        self.assertEqual(len(seg.spiketrains), 2)

        # Testing different parameter combinations
        seg = nio.read_segment(seg_index=0, load_waveforms=True)
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(len(seg.spiketrains), 2)
        self.assertTrue(len(seg.spiketrains[0].waveforms) > 0)
        self.assertTrue(len(seg.events) > 0)


class TestCheetah_v574(CommonNeuralynxIOTest, unittest.TestCase):
    cheetah_version = '5.7.4'
    files_to_test = []

    def test_read_block(self):
        dirname = self.get_local_path('neuralynx/Cheetah_v5.7.4/original_data')
        nio = NeuralynxIO(dirname=dirname, use_cache=False)

        block = nio.read_block()

        # Everything put in one segment
        seg = block.segments[0]
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(seg.analogsignals[0].shape[-1], 5)

        self.assertEqual(seg.analogsignals[0].sampling_rate, 32 * pq.kHz)
        self.assertEqual(len(seg.spiketrains), 0)  # no nse files available

        # Testing different parameter combinations
        block = nio.read_block(load_waveforms=True)
        self.assertEqual(len(block.segments[0].analogsignals), 1)
        self.assertEqual(len(block.segments[0].spiketrains), 0)
        self.assertGreater(len(block.segments[0].events), 0)

        block = nio.read_block(signal_group_mode='split-all')
        self.assertEqual(len(block.groups), 5)

        block = nio.read_block(signal_group_mode='group-by-same-units')
        self.assertEqual(len(block.groups), 1)

    def test_read_single_file(self):
        filename = self.get_local_path(
            'neuralynx/Cheetah_v5.7.4/original_data/CSC1.ncs'
        )
        nio = NeuralynxIO(filename=filename, use_cache=False)
        block = nio.read_block()
        self.assertTrue(len(block.segments[0].analogsignals) > 0)
        self.assertTrue((len(block.segments[0].spiketrains)) == 0)
        self.assertTrue((len(block.segments[0].events)) == 0)
        self.assertTrue((len(block.segments[0].epochs)) == 0)

    def test_exclude_filename(self):
        dname = self.get_local_path(
            'neuralynx/Cheetah_v5.7.4/original_data/'
        )

        # exclude a single file
        nio = NeuralynxIO(dirname=dname, exclude_filename='CSC1.ncs', use_cache=False)
        block = nio.read_block()
        self.assertTrue(len(block.segments[0].analogsignals) > 0)
        self.assertTrue((len(block.segments[0].spiketrains)) >= 0)
        self.assertTrue((len(block.segments[0].events)) >= 0)
        self.assertTrue((len(block.segments[0].epochs)) == 0)

        # exclude all ncs files from session
        exclude_files = [f'CSC{i}.ncs' for i in range(6)]
        nio = NeuralynxIO(dirname=dname, exclude_filename=exclude_files,
                          use_cache=False)
        block = nio.read_block()
        self.assertTrue(len(block.segments[0].analogsignals) == 0)
        self.assertTrue((len(block.segments[0].spiketrains)) >= 0)
        self.assertTrue((len(block.segments[0].events)) >= 0)
        self.assertTrue((len(block.segments[0].epochs)) == 0)


class TestPegasus_v211(CommonNeuralynxIOTest, unittest.TestCase):
    pegasus_version = '2.1.1'
    files_to_test = []

    def test_read_block(self):
        dirname = self.get_local_path('neuralynx/Pegasus_v2.1.1')
        nio = NeuralynxIO(dirname=dirname, use_cache=False)

        block = nio.read_block()

        # Everything put in one segment
        seg = block.segments[0]
        self.assertEqual(len(seg.analogsignals), 0)  # no ncs file available
        self.assertGreater(len(block.segments[0].events), 1)  # single nev file available
        self.assertEqual(len(seg.spiketrains), 0)  # no nse files available

        # Testing different parameter combinations
        block = nio.read_block(load_waveforms=True)
        self.assertEqual(len(block.segments[0].spiketrains), 0)
        self.assertGreater(len(block.segments[0].events), 1)

        block = nio.read_block(signal_group_mode='split-all')

        block = nio.read_block(signal_group_mode='group-by-same-units')


class TestData(CommonNeuralynxIOTest, unittest.TestCase):

    def _load_plaindata(self, filename, numSamps):
        """
        Load numSamps samples only from Ncs dump files which contain one row for each record,
        each row containing the timestamp, channel number, whole integer sampling frequency,
        number of samples, followed by that number of samples (which may be different for
        each record).
        """
        res = []
        totRes = 0
        with open(filename) as f:
            for line in f:
                vals = list(map(int, line.split()))
                numSampsThisLine = len(vals) - 4
                if numSampsThisLine < 0 or numSampsThisLine < vals[3]:
                    raise IOError('plain data file "' + filename + ' improperly formatted')
                numAvail = min(numSampsThisLine, vals[3])  # only use valid samples
                if numAvail < numSamps - len(res):
                    res.append(vals[4:(4 + numAvail)])
                else:
                    res.append(vals[4:(4 + numSamps - len(res))])
                totRes += len(res[-1])
                if totRes == numSamps:
                    break

            return [item for sublist in res for item in sublist]

    def test_ncs(self):
        for session in self.files_to_test:
            dirname = self.get_local_path(session)
            nio = NeuralynxIO(dirname=dirname, use_cache=False)
            block = nio.read_block()

            # check that data agrees in first segment first channel only
            for anasig_id, anasig in enumerate(block.segments[0].analogsignals):
                chid = anasig.array_annotations['channel_ids'][0]

                chname = str(anasig.array_annotations['channel_names'][0])
                chuid = (chname, chid)
                filename = nio.ncs_filenames[chuid][:-3] + 'txt'
                filename = filename.replace('original_data', 'plain_data')
                overlap = 512 * 500
                if os.path.isfile(filename):
                    plain_data = self._load_plaindata(filename, overlap)
                    gain_factor_0 = plain_data[0] / anasig.magnitude[0, 0]
                    numToTest = min(len(plain_data), len(anasig.magnitude[:, 0]))
                    np.testing.assert_allclose(plain_data[:numToTest],
                                               anasig.magnitude[:numToTest, 0] * gain_factor_0,
                                               rtol=0.01, err_msg=" for file " + filename)
                else:
                    warnings.warn(f'Could not find corresponding test file {filename}')
                    # TODO: Create missing plain data file using NeuraView
                    # https://neuralynx.com/software/category/data-analysis

    def test_keep_original_spike_times(self):
        for session in self.files_to_test:
            dirname = self.get_local_path(session)
            nio = NeuralynxIO(dirname=dirname, keep_original_times=True)
            block = nio.read_block()

            for st in block.segments[0].spiketrains:
                filename = st.file_origin.replace('original_data', 'plain_data')
                if '.nse' in st.file_origin:
                    filename = filename.replace('.nse', '.txt')
                    times_column = 0
                    plain_data = np.loadtxt(filename)[:, times_column]
                elif '.ntt' in st.file_origin:
                    filename = filename.replace('.ntt', '.txt')
                    times_column = 2
                    plain_data = np.loadtxt(filename)[:, times_column]
                    # ntt files contain 4 rows per spike time
                    plain_data = plain_data[::4]

                times = st.rescale(pq.microsecond).magnitude
                overlap = min(len(plain_data), len(times))
                np.testing.assert_allclose(plain_data[:overlap], times[:overlap], rtol=1e-10)


class TestIncompleteBlocks(CommonNeuralynxIOTest, unittest.TestCase):
    def test_incomplete_block_handling_v632(self):
        dirname = self.get_local_path('neuralynx/Cheetah_v6.3.2/incomplete_blocks')
        nio = NeuralynxIO(dirname=dirname, use_cache=False)

        block = nio.read_block()

        # known gap values
        n_gaps = 2
        # so 3 segments, 3 anasigs by Channelindex
        self.assertEqual(len(block.segments), n_gaps + 1)
        # self.assertEqual(len(block.channel_indexes[0].analogsignals), n_gaps + 1)

        expected_segment_starts = [8124.582909, 8427.832053, 8487.768561]
        expected_segment_stops = [8427.831990, 8487.768498, 10794.133994]
        for seg_idx in range(n_gaps + 1):

            t_start = nio.segment_t_start(0, seg_idx) + nio.global_t_start
            t_stop = nio.segment_t_stop(0, seg_idx) + nio.global_t_start

            self.assertEqual(np.round(t_start, 4), np.round(expected_segment_starts[seg_idx], 4))
            self.assertEqual(np.round(t_stop, 4), np.round(expected_segment_stops[seg_idx], 4))


class TestGaps(CommonNeuralynxIOTest, unittest.TestCase):
    def test_gap_handling_v551(self):
        dirname = self.get_local_path('neuralynx/Cheetah_v5.5.1/original_data')
        nio = NeuralynxIO(dirname=dirname, use_cache=False)

        block = nio.read_block()

        # known gap values
        n_gaps = 1
        # so 2 segments, 2 anasigs by Channelindex, 2 SpikeTrain by Units
        self.assertEqual(len(block.segments), n_gaps + 1)
        # self.assertEqual(len(block.channel_indexes[0].analogsignals), n_gaps + 1)
        # self.assertEqual(len(block.channel_indexes[-1].units[0].spiketrains), n_gaps + 1)

    def test_gap_handling_v563(self):
        dirname = self.get_local_path('neuralynx/Cheetah_v5.6.3/original_data')
        nio = NeuralynxIO(dirname=dirname, use_cache=False)
        block = nio.read_block()

        # known gap values
        n_gaps = 1
        # so 2 segments, 2 anasigs by Channelindex, 2 SpikeTrain by Units
        self.assertEqual(len(block.segments), n_gaps + 1)
        # self.assertEqual(len(block.channel_indexes[0].analogsignals), n_gaps + 1)
        # self.assertEqual(len(block.channel_indexes[-1].units[0].spiketrains), n_gaps + 1)


class TestMultiSamplingRates(CommonNeuralynxIOTest, unittest.TestCase):
    def test_multi_sampling_rates(self):
        # Test Cheetah 6.4.1, with different sampling rates across ncs files.
        nio = NeuralynxIO(self.get_local_path('neuralynx/Cheetah_v6.4.1dev/original_data'))
        block = nio.read_block()

        self.assertEqual(len(block.segments), 1)
        seg = block.segments[0]
        self.assertEqual(len(seg.analogsignals), 3)
        self.assertEqual(len(np.unique([sig.sampling_rate for sig in seg.analogsignals])), 3)

        # check if the sampling rate of analogsignals are correct
        expected_rates = [2, 2.66667, 32] * pq.kHz
        for sig in block.segments[0].analogsignals:
            observed_rate = sig.sampling_rate.rescale(expected_rates.units)
            # using isclose for flexible float comparison
            self.assertTrue(any([np.isclose(observed_rate, exp_rate)]) for
                            exp_rate in expected_rates)


def compare_neo_content(bl1, bl2):
    print('*' * 5, 'Comparison of blocks', '*' * 5)
    object_types_to_test = [Segment, AnalogSignal,
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
