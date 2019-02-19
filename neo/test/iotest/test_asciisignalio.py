# -*- coding: utf-8 -*-
"""
Tests of neo.io.asciisignalio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

import os
import unittest
import json
import csv
import numpy as np
import quantities as pq
from numpy.testing import assert_array_almost_equal, assert_array_equal
from neo.io import AsciiSignalIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.core import AnalogSignal, Segment, Block


class TestAsciiSignalIOWithTestFiles(BaseTestIO, unittest.TestCase):
    ioclass = AsciiSignalIO
    files_to_download = [  # 'File_asciisignal_1.asc',
        'File_asciisignal_2.txt',
        'File_asciisignal_3.txt',
    ]
    files_to_test = files_to_download


class TestAsciiSignalIO(unittest.TestCase):

    def test_genfromtxt_expect_success(self):
        sample_data = np.random.uniform(size=(200, 3))
        filename = "test_genfromtxt_expect_success.txt"
        np.savetxt(filename, sample_data, delimiter=' ')
        sampling_rate = 1 * pq.kHz
        io = AsciiSignalIO(filename, sampling_rate=sampling_rate, delimiter=' ',
                           units='mV', method='genfromtxt')
        block = io.read_block()

        signal1 = block.segments[0].analogsignals[1]
        assert_array_almost_equal(signal1.reshape(-1).magnitude, sample_data[:, 1],
                                  decimal=6)
        self.assertEqual(len(block.segments[0].analogsignals), 3)
        self.assertEqual(signal1.t_stop, sample_data.shape[0] / sampling_rate)
        self.assertEqual(signal1.units, pq.mV)

        os.remove(filename)

    # test_genfromtxt_expect_failure

    def test_csv_expect_success(self):
        filename = 'test_csv_expect_success.csv'
        sample_data = [
            (-65, -65, -65, 0.5),
            (-64.8, -64.5, -64.0, 0.6),
            (-64.6, -64.2, -77.0, 0.7),
            (-64.3, -64.0, -99.9, 0.8)
        ]
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in sample_data:
                writer.writerow(row)

        io = AsciiSignalIO(filename, usecols=(0, 1, 3), timecolumn=2,
                           # note that timecolumn applies to the remaining columns
                           # after applying usecols
                           time_units="ms", delimiter=',', units="mV", method='csv',
                           signal_group_mode='all-in-one', t_start=0.5)

        block = io.read_block()
        signal = block.segments[0].analogsignals[0]
        self.assertEqual(signal.shape, (4, 2))  # two columns remaining after usecols
                                                # and timecolumn applied
        assert_array_almost_equal(signal[:, 1].reshape(-1).magnitude,
                                  np.array(sample_data)[:, 1],
                                  decimal=5)
        self.assertAlmostEqual(signal.sampling_period, 0.1 * pq.ms)

        os.remove(filename)
    # test_csv_expect_failure
    # test_homemade_expect_success

    def test_homemade_expect_success(self):
        filename = 'test_homemade_expect_success.txt'
        sample_data = [
            (-65, -65, -65, 0.5),
            (-64.8, -64.5, -64.0, 0.6),
            (-64.6, -64.2, -77.0, 0.7),
            (-64.3, -64.0, -99.9, 0.8)
        ]
        with open(filename, 'w') as datafile:
            datafile.write("# a comment\n")
            for row in sample_data:
                datafile.write("\t ".join(map(str, row)) + "\t\n")

        io = AsciiSignalIO(filename, usecols=(0, 1, 3), timecolumn=2, skiprows=1,
                           time_units="ms", delimiter='\t', units="mV", method='homemade',
                           signal_group_mode='all-in-one', t_start=0.5)

        block = io.read_block()
        signal = block.segments[0].analogsignals[0]
        self.assertEqual(signal.shape, (4, 2))  # two columns remaining after usecols
                                                # and timecolumn applied
        assert_array_almost_equal(signal[:, 1].reshape(-1).magnitude,
                                  np.array(sample_data)[:, 1],
                                  decimal=5)
        self.assertAlmostEqual(signal.sampling_period, 0.1 * pq.ms)

        os.remove(filename)

    # test_homemade_expect_failure

    def test_callable_expect_success(self):
        sample_data = np.random.uniform(size=(200, 3))
        filename = "test_genfromtxt_expect_success.txt"
        np.savetxt(filename, sample_data, delimiter=' ')
        sampling_rate = 1 * pq.kHz

        def reader(filename, comment_rows):
            return np.genfromtxt(filename, delimiter=' ', usecols=None,
                                 skip_header=comment_rows or 0, dtype='f')

        io = AsciiSignalIO(filename, sampling_rate=sampling_rate, delimiter=' ',
                           units='mV', method=reader)
        block = io.read_block()

        signal1 = block.segments[0].analogsignals[1]
        assert_array_almost_equal(signal1.reshape(-1).magnitude, sample_data[:, 1],
                                  decimal=6)
        self.assertEqual(len(block.segments[0].analogsignals), 3)
        self.assertEqual(signal1.t_stop, sample_data.shape[0] / sampling_rate)
        self.assertEqual(signal1.units, pq.mV)

        os.remove(filename)

    # test usecols
    # test skiprows

    def test_timecolumn(self):
        sample_data = np.random.uniform(size=(200, 3))
        sampling_period = 0.5
        time_data = sampling_period * np.arange(sample_data.shape[0])
        combined_data = np.hstack((sample_data, time_data[:, np.newaxis]))
        filename = "test_multichannel.txt"
        np.savetxt(filename, combined_data, delimiter=' ')
        io = AsciiSignalIO(filename, delimiter=' ',
                           units='mV', method='genfromtxt', timecolumn=-1,
                           time_units='ms', signal_group_mode='split-all')
        block = io.read_block()

        signal1 = block.segments[0].analogsignals[1]
        assert_array_almost_equal(signal1.reshape(-1).magnitude, sample_data[:, 1],
                                  decimal=6)
        self.assertEqual(signal1.sampling_period, sampling_period * pq.ms)
        self.assertEqual(len(block.segments[0].analogsignals), 3)
        self.assertEqual(signal1.t_stop, sample_data.shape[0] * sampling_period * pq.ms)
        self.assertEqual(signal1.units, pq.mV)

        os.remove(filename)

    def test_multichannel(self):
        sample_data = np.random.uniform(size=(200, 3))
        filename = "test_multichannel.txt"
        np.savetxt(filename, sample_data, delimiter=' ')
        sampling_rate = 1 * pq.kHz
        io = AsciiSignalIO(filename, sampling_rate=sampling_rate, delimiter=' ',
                           units='mV', method='genfromtxt',
                           signal_group_mode='all-in-one')
        block = io.read_block()

        signal = block.segments[0].analogsignals[0]
        assert_array_almost_equal(signal.magnitude, sample_data,
                                  decimal=6)
        self.assertEqual(len(block.segments[0].analogsignals), 1)
        self.assertEqual(signal.t_stop, sample_data.shape[0] / sampling_rate)
        self.assertEqual(signal.units, pq.mV)

        os.remove(filename)

    def test_multichannel_with_timecolumn(self):
        sample_data = np.random.uniform(size=(200, 3))
        sampling_period = 0.5
        time_data = sampling_period * np.arange(sample_data.shape[0])
        combined_data = np.hstack((time_data[:, np.newaxis], sample_data))
        filename = "test_multichannel.txt"
        np.savetxt(filename, combined_data, delimiter=' ')
        io = AsciiSignalIO(filename, delimiter=' ',
                           units='mV', method='genfromtxt', timecolumn=0,
                           time_units='ms',
                           signal_group_mode='all-in-one')
        block = io.read_block()

        signal = block.segments[0].analogsignals[0]
        assert_array_almost_equal(signal.magnitude, sample_data,
                                  decimal=6)
        self.assertEqual(signal.sampling_period, sampling_period * pq.ms)
        self.assertEqual(len(block.segments[0].analogsignals), 1)
        self.assertEqual(signal.t_stop, sample_data.shape[0] * sampling_period * pq.ms)
        self.assertEqual(signal.units, pq.mV)

        os.remove(filename)

    def test_multichannel_with_negative_timecolumn(self):
        sample_data = np.random.uniform(size=(200, 3))
        sampling_period = 0.5
        time_data = sampling_period * np.arange(sample_data.shape[0])
        combined_data = np.hstack((sample_data, time_data[:, np.newaxis]))
        filename = "test_multichannel.txt"
        np.savetxt(filename, combined_data, delimiter=' ')
        io = AsciiSignalIO(filename, delimiter=' ',
                           units='mV', method='genfromtxt', timecolumn=-1,
                           time_units='ms',
                           signal_group_mode='all-in-one')
        block = io.read_block()

        signal = block.segments[0].analogsignals[0]
        assert_array_almost_equal(signal.magnitude, sample_data,
                                  decimal=6)
        self.assertEqual(signal.sampling_period, sampling_period * pq.ms)
        self.assertEqual(len(block.segments[0].analogsignals), 1)
        self.assertEqual(signal.t_stop, sample_data.shape[0] * sampling_period * pq.ms)
        self.assertEqual(signal.units, pq.mV)

        os.remove(filename)

    # test write without timecolumn
    # test write with timecolumn
    # test write with units/timeunits different from those of signal

    def test_read_with_json_metadata(self):
        sample_data = np.random.uniform(size=(200, 3))
        filename = "test_read_with_json_metadata.txt"
        metadata_filename = "test_read_with_json_metadata_about.json"
        np.savetxt(filename, sample_data, delimiter=' ')

        metadata = {
            "filename": filename,
            "delimiter": " ",
            "timecolumn": None,
            "units": "mV",
            "time_units": "ms",
            "sampling_rate": {
                "value": 1.0,
                "units": "kHz"
            },
            "method": "genfromtxt",
            "signal_group_mode": 'split-all'
        }
        with open(metadata_filename, "w") as fp:
            json.dump(metadata, fp)

        expected_sampling_rate = 1 * pq.kHz
        io = AsciiSignalIO(filename)
        block = io.read_block()

        signal1 = block.segments[0].analogsignals[1]
        assert_array_almost_equal(signal1.reshape(-1).magnitude, sample_data[:, 1],
                                  decimal=6)
        self.assertEqual(len(block.segments[0].analogsignals), 3)
        self.assertEqual(signal1.sampling_rate, expected_sampling_rate)
        self.assertEqual(signal1.t_stop, sample_data.shape[0] / signal1.sampling_rate)
        self.assertEqual(signal1.units, pq.mV)

        os.remove(filename)
        os.remove(metadata_filename)

    def test_roundtrip_with_json_metadata(self):
        sample_data = np.random.uniform(size=(200, 3))
        filename = "test_roundtrip_with_json_metadata.txt"
        metadata_filename = "test_roundtrip_with_json_metadata_about.json"
        signal1 = AnalogSignal(sample_data, units="pA", sampling_rate=2 * pq.kHz)
        seg1 = Segment()
        block1 = Block()
        seg1.analogsignals.append(signal1)
        seg1.block = block1
        block1.segments.append(seg1)

        iow = AsciiSignalIO(filename, metadata_filename=metadata_filename)
        iow.write_block(block1)
        self.assert_(os.path.exists(metadata_filename))

        ior = AsciiSignalIO(filename)
        block2 = ior.read_block()
        assert len(block2.segments[0].analogsignals) == 3
        signal2 = block2.segments[0].analogsignals[1]

        assert_array_almost_equal(signal1.magnitude[:, 1], signal2.magnitude.reshape(-1),
                                  decimal=7)
        self.assertEqual(signal1.units, signal2.units)
        self.assertEqual(signal1.sampling_rate, signal2.sampling_rate)
        assert_array_equal(signal1.times, signal2.times)

        os.remove(filename)
        os.remove(metadata_filename)

    def test_genfromtxt_irregular_expect_success(self):
        sample_data = np.random.uniform(size=(200, 3))
        sample_data[:, 0] = np.sort(sample_data[:, 0])  # make column 0 the time column
        filename = "test_genfromtxt_irregular_expect_success.txt"
        np.savetxt(filename, sample_data, delimiter=' ')

        io = AsciiSignalIO(filename, delimiter=' ', timecolumn=0,
                           units='mV', method='genfromtxt', signal_group_mode='split-all')
        block = io.read_block()

        signal1 = block.segments[0].irregularlysampledsignals[1]
        assert_array_almost_equal(signal1.reshape(-1).magnitude, sample_data[:, 2],
                                  decimal=6)
        self.assertEqual(len(block.segments[0].analogsignals), 0)
        self.assertEqual(len(block.segments[0].irregularlysampledsignals), 2)
        self.assertEqual(signal1.units, pq.mV)

        os.remove(filename)

    def test_irregular_multichannel(self):
        sample_data = np.random.uniform(size=(200, 3))
        sample_data[:, 0] = np.sort(sample_data[:, 0])  # make column 0 the time column
        filename = "test_irregular_multichannel.txt"
        np.savetxt(filename, sample_data, delimiter=' ')

        io = AsciiSignalIO(filename, delimiter=' ', timecolumn=0,
                           units='mV', method='genfromtxt', signal_group_mode='all-in-one')
        block = io.read_block()

        signal = block.segments[0].irregularlysampledsignals[0]
        assert_array_almost_equal(signal.magnitude, sample_data[:, 1:3],
                                  decimal=6)
        self.assertEqual(len(block.segments[0].analogsignals), 0)
        self.assertEqual(len(block.segments[0].irregularlysampledsignals), 1)
        self.assertEqual(signal.shape, (200, 2))
        self.assertEqual(signal.units, pq.mV)

        os.remove(filename)


if __name__ == "__main__":
    unittest.main()
