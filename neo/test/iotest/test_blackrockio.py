# -*- coding: utf-8 -*-
"""
Tests of neo.io.blackrockio
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import os
import struct
import sys
import tempfile

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from numpy.testing import assert_equal

import numpy as np
import quantities as pq

from neo.io.blackrockio import BlackrockIO

from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.iotest.tools import get_test_file_full_path

# check scipy
try:
    from distutils import version
    import scipy.io
    import scipy.version
except ImportError as err:
    HAVE_SCIPY = False
    SCIPY_ERR = err
else:
    if version.LooseVersion(scipy.version.version) < '0.8':
        HAVE_SCIPY = False
        SCIPY_ERR = ImportError("your scipy version is too old to support " +
                                "MatlabIO, you need at least 0.8. " +
                                "You have %s" % scipy.version.version)
    else:
        HAVE_SCIPY = True
        SCIPY_ERR = None

class CommonTests(BaseTestIO, unittest.TestCase):
    ioclass =BlackrockIO

    files_to_test = [
        #'test2/test.ns5'
        ]

    files_to_download = [
        #'test2/test.ns5'
        ]

    files_to_test = ['FileSpec2.3001']

    files_to_download = [
        'FileSpec2.3001.nev',
        'FileSpec2.3001.ns5',
        'FileSpec2.3001.ccf',
        'FileSpec2.3001.mat']

    ioclass = BlackrockIO

    def test_inputs_V23(self):
        """
        Test various inputs to BlackrockIO.read_block with version 2.3 file
        to check for parsing errors.
        """

        try:
            b = BlackrockIO(
                get_test_file_full_path(
                    ioclass=BlackrockIO,
                    filename='FileSpec2.3001',
                    directory=self.local_test_dir, clean=False),
                verbose=False)

        except:
            self.fail()

        # Load data to maximum extent, one None is not given as list
        block = b.read_block(
            n_starts=[None], n_stops=None, channels=range(1, 9),
            nsx_to_load=5, units='all', load_events=True,
            load_waveforms=False)
        lena = len(block.segments[0].analogsignals[0])
        numspa = len(block.segments[0].spiketrains[0])

        # Load data using a negative time and a time exceeding the end of the
        # recording
        too_large_tstop = block.segments[0].analogsignals[0].t_stop + 1 * pq.s
        block = b.read_block(
            n_starts=[-100 * pq.ms], n_stops=[too_large_tstop],
            channels=range(1, 9), nsx_to_load=[5], units='all',
            load_events=False, load_waveforms=False)
        lenb = len(block.segments[0].analogsignals[0])
        numspb = len(block.segments[0].spiketrains[0])

        # Same length of analog signal?
        # Both should have read the complete data set!
        self.assertEqual(lena, lenb)

        # Same length of spike train?
        # Both should have read the complete data set!
        self.assertEqual(numspa, numspb)

        # n_starts and n_stops not given as list
        # verifies identical length of returned signals given equal durations
        # as input
        ns5_unit = block.segments[0].analogsignals[0].sampling_period
        block = b.read_block(
            n_starts=100 * ns5_unit, n_stops=200 * ns5_unit,
            channels=range(1, 9), nsx_to_load=5, units='all',
            load_events=False, load_waveforms=False)
        lena = len(block.segments[0].analogsignals[0])

        block = b.read_block(
            n_starts=301 * ns5_unit, n_stops=401 * ns5_unit,
            channels=range(1, 9), nsx_to_load=5, units='all',
            load_events=False, load_waveforms=False)
        lenb = len(block.segments[0].analogsignals[0])

        # Same length?
        self.assertEqual(lena, lenb)
        # Length should be 100 samples exactly
        self.assertEqual(lena, 100)

        # Load partial data types and check if this is selection is made
        block = b.read_block(
            n_starts=None, n_stops=None, channels=range(1, 9),
            nsx_to_load=5, units='none', load_events=False,
            load_waveforms=True)

        self.assertEqual(len(block.segments), 1)
        self.assertEqual(len(block.segments[0].analogsignals), 8)
        self.assertEqual(len(block.channel_indexes), 8)
        self.assertEqual(len(block.channel_indexes[0].units), 0)
        self.assertEqual(len(block.segments[0].events), 0)
        self.assertEqual(len(block.segments[0].spiketrains), 0)

        # NOTE: channel 6 does not contain any unit
        block = b.read_block(
            n_starts=[None, 3000 * pq.ms], n_stops=[1000 * pq.ms, None],
            channels=range(1, 9), nsx_to_load='none',
            units={1: 0, 5: 0, 6: 0}, load_events=True,
            load_waveforms=True)

        self.assertEqual(len(block.segments), 2)
        self.assertEqual(len(block.segments[0].analogsignals), 0)
        self.assertEqual(len(block.channel_indexes), 8)
        self.assertEqual(len(block.channel_indexes[0].units), 1)
        self.assertEqual(len(block.segments[0].events), 0)
        self.assertEqual(len(block.segments[0].spiketrains), 2)

    @unittest.skipUnless(HAVE_SCIPY, "requires scipy")
    def test_compare_blackrockio_with_matlabloader(self):
        """
        This test compares the output of ReachGraspIO.read_block() with the
        output generated by a Matlab implementation of a Blackrock file reader
        provided by the company. The output for comparison is provided in a
        .mat file created by the script create_data_matlab_blackrock.m.
        The function tests LFPs, spike times, and digital events on channels
        80-83 and spike waveforms on channel 82, unit 1.
        For details on the file contents, refer to FileSpec2.3.txt
        """

        # Load data from Matlab generated files
        ml = scipy.io.loadmat(
            get_test_file_full_path(
                ioclass=BlackrockIO,
                filename='FileSpec2.3001.mat',
                directory=self.local_test_dir, clean=False))
        lfp_ml = ml['lfp']  # (channel x time) LFP matrix
        ts_ml = ml['ts']  # spike time stamps
        elec_ml = ml['el']  # spike electrodes
        unit_ml = ml['un']  # spike unit IDs
        wf_ml = ml['wf']  # waveform unit 1 channel 1
        mts_ml = ml['mts']  # marker time stamps
        mid_ml = ml['mid']  # marker IDs

        # Load data in channels 1-3 from original data files using the neo
        # framework
        session = BlackrockIO(
            get_test_file_full_path(
                ioclass=BlackrockIO,
                filename='FileSpec2.3001',
                directory=self.local_test_dir, clean=False),
            verbose=False)
        block = session.read_block(load_waveforms=True)

        # Check if analog data on channels 1-8 are equal
        for rcg_i in block.channel_indexes:
            # Should only have one recording channel per group
            self.assertEqual(rcg_i.size, 1)

            idx = rcg_i[0]
            if idx in range(1, 9):
                assert_equal(rcg_i.analogsignal.base, lfp_ml[idx - 1, :])

        # Should only have one segment
        self.assertEqual(len(block.segments), 1)

        # Check if spikes in channels 1,3,5,7 are equal
        for st_i in block.segments[0].spiketrains:
            channelid = st_i.annotations['channel_id']
            if channelid in range(1, 7, 2):
                unitid = st_i.annotations['unit_id']
                matlab_spikes = ts_ml[np.nonzero(
                    np.logical_and(elec_ml == channelid, unit_ml == unitid))]
                assert_equal(st_i.base, matlab_spikes)

                # Check waveforms of channel 1, unit 0
                if channelid == 1 and unitid == 0:
                    assert_equal(st_i.waveforms, wf_ml)

        # Check if digital marker events are equal
        for ea_i in block.segments[0].events:
            if ('digital_marker' in ea_i.annotations.keys()) and (
                    ea_i.annotations['digital_marker'] is True):
                markerid = ea_i.annotations['marker_id']
                matlab_digievents = mts_ml[np.nonzero(mid_ml == markerid)]
                assert_equal(ea_i.times.base, matlab_digievents)

        # Check if analog marker events are equal
        # Currently not implemented by the Matlab loader
        for ea_i in block.segments[0].events:
            if ('analog_marker' in ea_i.annotations.keys()) and (
                    ea_i.annotations['analog_marker'] is True):
                markerid = ea_i.annotations['marker_id']
                matlab_anaevents = mts_ml[np.nonzero(mid_ml == markerid)]
                assert_equal(ea_i.times.base, matlab_anaevents)


if __name__ == '__main__':
    unittest.main()
