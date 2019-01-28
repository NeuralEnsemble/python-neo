# -*- coding: utf-8 -*-
"""
Tests of neo.io.asciisignalio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

import os
import unittest
import numpy as np
import quantities as pq
from numpy.testing import assert_array_almost_equal
from neo.io import AsciiSignalIO
from neo.test.iotest.common_io_test import BaseTestIO


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
    # test_csv_expect_success
    # test_csv_expect_failure
    # test_homemade_expect_success
    # test_homemade_expect_failure
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
                           time_units='ms', multichannel=False)
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
                           multichannel=True)
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
                           multichannel=True)
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
                           multichannel=True)
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


if __name__ == "__main__":
    unittest.main()
