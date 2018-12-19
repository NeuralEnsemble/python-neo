# -*- coding: utf-8 -*-
"""
Tests of neo.io.blackrockio
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import unittest
import warnings

from numpy.testing import assert_equal

import numpy as np
import quantities as pq

from neo.io.blackrockio import BlackrockIO

from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.iotest.tools import get_test_file_full_path
from neo.test.tools import assert_neo_object_is_compliant

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
    ioclass = BlackrockIO

    files_to_test = ['FileSpec2.3001',
        'blackrock_2_1/l101210-001']

    files_to_download = [
        'FileSpec2.3001.nev',
        'FileSpec2.3001.ns5',
        'FileSpec2.3001.ccf',
        'FileSpec2.3001.mat',
        'blackrock_2_1/l101210-001.mat',
        'blackrock_2_1/l101210-001_nev-02_ns5.mat',
        'blackrock_2_1/l101210-001.ns2',
        'blackrock_2_1/l101210-001.ns5',
        'blackrock_2_1/l101210-001.nev',
        'blackrock_2_1/l101210-001-02.nev',
        'segment/PauseCorrect/pause_correct.nev',
        'segment/PauseCorrect/pause_correct.ns2',
        'segment/PauseSpikesOutside/pause_spikes_outside_seg.nev',
        'segment/ResetCorrect/reset.nev',
        'segment/ResetCorrect/reset.ns2',
        'segment/ResetFail/reset_fail.nev']

    ioclass = BlackrockIO

    def test_load_waveforms(self):
        filename = self.get_filename_path('FileSpec2.3001')
        reader = BlackrockIO(filename=filename, verbose=False)

        bl = reader.read_block(load_waveforms=True)
        assert_neo_object_is_compliant(bl)

    def test_inputs_V23(self):
        """
        Test various inputs to BlackrockIO.read_block with version 2.3 file
        to check for parsing errors.
        """
        filename = self.get_filename_path('FileSpec2.3001')
        reader = BlackrockIO(filename=filename, verbose=False, nsx_to_load=5)

        # Assert IOError is raised when no Blackrock files are available
        with self.assertRaises(IOError):
            reader2 = BlackrockIO(filename='nonexistent')

        # Load data to maximum extent, one None is not given as list
        block = reader.read_block(load_waveforms=False)
        lena = len(block.segments[0].analogsignals[0])
        numspa = len(block.segments[0].spiketrains[0])

        # Load data using a negative time and a time exceeding the end of the
        # recording
        too_large_tstop = block.segments[0].analogsignals[0].t_stop + 1 * pq.s
        buggy_slice = (-100 * pq.ms, too_large_tstop)

        # this is valid in read_segment because seg_index is specified
        seg = reader.read_segment(seg_index=0, time_slice=buggy_slice)

        lenb = len(seg.analogsignals[0])
        numspb = len(seg.spiketrains[0])

        # Same length of analog signal?
        # Both should have read the complete data set!
        self.assertEqual(lena, lenb)

        # Same length of spike train?
        # Both should have read the complete data set!
        self.assertEqual(numspa, numspb)

        # test 4 Units
        block = reader.read_block(load_waveforms=True,
                                signal_group_mode='split-all',
                                units_group_mode='all-in-one')

        self.assertEqual(len(block.segments[0].analogsignals), 10)
        self.assertEqual(len(block.channel_indexes[-1].units), 4)
        self.assertEqual(len(block.channel_indexes[-1].units),
                         len(block.segments[0].spiketrains))

        anasig = block.segments[0].analogsignals[0]
        self.assertIsNotNone(anasig.file_origin)

    def test_inputs_V21(self):
        """
        Test various inputs to BlackrockIO.read_block with version 2.3 file
        to check for parsing errors.
        """
        filename = self.get_filename_path('blackrock_2_1/l101210-001')
        reader = BlackrockIO(filename=filename, verbose=False, nsx_to_load=5)

        # Assert IOError is raised when no Blackrock files are available
        with self.assertRaises(IOError):
            reader2 = BlackrockIO(filename='nonexistent')
        # with self.assertRaises(IOError):
        #     reader2 = BlackrockIO(filename=filename, nev_override='nonexistent')

        # Load data to maximum extent, one None is not given as list
        block = reader.read_block(load_waveforms=False, signal_group_mode='split-all')
        lena = len(block.segments[0].analogsignals[0])
        numspa = len(block.segments[0].spiketrains[0])

        # Load data using a negative time and a time exceeding the end of the
        # recording
        too_large_tstop = block.segments[0].analogsignals[0].t_stop + 1 * pq.s
        buggy_slice = (-100 * pq.ms, too_large_tstop)

        # This is valid in read_segment because seg_index is specified
        seg = reader.read_segment(seg_index=0, time_slice=buggy_slice)

        lenb = len(seg.analogsignals[0])
        numspb = len(seg.spiketrains[0])

        # Same length of analog signal?
        # Both should have read the complete data set!
        self.assertEqual(lena, lenb)

        # Same length of spike train?
        # Both should have read the complete data set!
        self.assertEqual(numspa, numspb)

        # test 4 Units
        block = reader.read_block(load_waveforms=True,
                                signal_group_mode='split-all',
                              units_group_mode='all-in-one')

        self.assertEqual(len(block.segments[0].analogsignals), 96)
        self.assertEqual(len(block.channel_indexes[-1].units), 218)
        self.assertEqual(len(block.channel_indexes[-1].units),
                         len(block.segments[0].spiketrains))

        anasig = block.segments[0].analogsignals[0]
        self.assertIsNotNone(anasig.file_origin)

    def test_load_muliple_nsx(self):
        """
        Test if multiple nsx signals can be loaded at the same time.
        """
        filename = self.get_filename_path('blackrock_2_1/l101210-001')
        reader = BlackrockIO(filename=filename, verbose=False, nsx_to_load='all')

        # number of different sampling rates corresponds to number of nsx signals, because
        # single nsx contains only signals of identical sampling rate
        block = reader.read_block(load_waveforms=False)
        sampling_rates = np.unique(
            [a.sampling_rate.rescale('Hz') for a in block.filter(objects='AnalogSignal')])
        self.assertEqual(len(sampling_rates), 2)

        segment = reader.read_segment()
        sampling_rates = np.unique(
            [a.sampling_rate.rescale('Hz') for a in segment.filter(objects='AnalogSignal')])
        self.assertEqual(len(sampling_rates), 2)

        # load only ns5
        reader = BlackrockIO(filename=filename, nsx_to_load=5)
        seg = reader.read_segment()
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(seg.analogsignals[0].shape, (109224, 96))

        # load only ns2
        reader = BlackrockIO(filename=filename, nsx_to_load=2)
        seg = reader.read_segment()
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(seg.analogsignals[0].shape, (3640, 6))

        # load only ns2
        reader = BlackrockIO(filename=filename, nsx_to_load=[2])
        seg = reader.read_segment()
        self.assertEqual(len(seg.analogsignals), 1)

        # load ns2 + ns5
        reader = BlackrockIO(filename=filename, nsx_to_load=[2, 5])
        seg = reader.read_segment()
        self.assertEqual(len(seg.analogsignals), 2)
        self.assertEqual(seg.analogsignals[0].shape, (3640, 6))
        self.assertEqual(seg.analogsignals[1].shape, (109224, 96))

        # load only ns5
        reader = BlackrockIO(filename=filename, nsx_to_load='max')
        seg = reader.read_segment()
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(seg.analogsignals[0].shape, (109224, 96))

    @unittest.skipUnless(HAVE_SCIPY, "requires scipy")
    def test_compare_blackrockio_with_matlabloader_v21(self):
        """
        This test compares the output of BlackrockIO.read_block() with the
        output generated by a Matlab implementation of a Blackrock file reader
        provided by the company. The output for comparison is provided in a
        .mat file created by the script create_data_matlab_blackrock.m.
        The function tests LFPs, spike times, and digital events.
        """

        dirname = get_test_file_full_path(ioclass=BlackrockIO,
                                          filename='blackrock_2_1/l101210-001',
                                          directory=self.local_test_dir, clean=False)
        # First run with parameters for ns5, then run with correct parameters for ns2
        parameters = [('blackrock_2_1/l101210-001_nev-02_ns5.mat',
                       {'nsx_to_load': 5, 'nev_override': '-'.join([dirname, '02'])}),
                      ('blackrock_2_1/l101210-001.mat', {'nsx_to_load': 2})]
        for index, param in enumerate(parameters):
            # Load data from matlab generated files
            ml = scipy.io.loadmat(
                get_test_file_full_path(
                    ioclass=BlackrockIO,
                    filename=param[0],
                    directory=self.local_test_dir, clean=False))
            lfp_ml = ml['lfp']  # (channel x time) LFP matrix
            ts_ml = ml['ts']  # spike time stamps
            elec_ml = ml['el']  # spike electrodes
            unit_ml = ml['un']  # spike unit IDs
            wf_ml = ml['wf']  # waveforms
            mts_ml = ml['mts']  # marker time stamps
            mid_ml = ml['mid']  # marker IDs

            # Load data from original data files using the Neo BlackrockIO
            session = BlackrockIO(
                dirname,
                verbose=False, **param[1])
            block = session.read_block(load_waveforms=True, signal_group_mode='split-all')
            # Check if analog data are equal
            self.assertGreater(len(block.channel_indexes), 0)
            for i, chidx in enumerate(block.channel_indexes):
                # Break for ChannelIndexes for Units that don't contain any Analogsignals
                if len(chidx.analogsignals) == 0 and len(chidx.units) >= 1:
                    break
                # Should only have one AnalogSignal per ChannelIndex
                self.assertEqual(len(chidx.analogsignals), 1)

                # Find out channel_id in order to compare correctly
                idx = chidx.analogsignals[0].annotations['channel_id']
                # Get data of AnalogSignal without pq.units
                anasig = np.squeeze(chidx.analogsignals[0].base[:].magnitude)
                # Test for equality of first nonzero values of AnalogSignal
                #                                   and matlab file contents
                # If not equal test if hardcoded gain is responsible for this
                # See BlackrockRawIO ll. 1420 commit 77a645655605ae39eca2de3ee511f3b522f11bd7
                j = 0
                while anasig[j] == 0:
                    j += 1
                if lfp_ml[i, j] != np.squeeze(chidx.analogsignals[0].base[j].magnitude):
                    anasig = anasig / 152.592547
                    anasig = np.round(anasig).astype(int)

                # Special case because id 142 is not included in ns2 file
                if idx == 143:
                    idx -= 1
                if idx > 128:
                    idx = idx - 136

                assert_equal(anasig, lfp_ml[idx - 1, :])

            # Check if spikes are equal
            self.assertEqual(len(block.segments), 1)
            for st_i in block.segments[0].spiketrains:
                channelid = st_i.annotations['channel_id']
                unitid = st_i.annotations['unit_id']

                # Compare waveforms
                matlab_wf = wf_ml[np.nonzero(
                    np.logical_and(elec_ml == channelid, unit_ml == unitid)), :][0]
                # Atleast_2d as correction for waveforms that are saved
                # in single dimension in SpikeTrain
                # because only one waveform is available
                assert_equal(np.atleast_2d(np.squeeze(st_i.waveforms).magnitude), matlab_wf)

                # Compare spike timestamps
                matlab_spikes = ts_ml[np.nonzero(
                    np.logical_and(elec_ml == channelid, unit_ml == unitid))]
                # Going sure that unit is really seconds and not 1/30000 seconds
                if (not st_i.units == pq.CompoundUnit("1.0/{0} * s".format(30000))) and \
                        st_i.units == pq.s:
                    st_i = np.round(st_i.base * 30000).astype(int)
                assert_equal(st_i, matlab_spikes)

            # Check if digital input port events are equal
            self.assertGreater(len(block.segments[0].events), 0)
            for ea_i in block.segments[0].events:
                if ea_i.name == 'digital_input_port':
                    # Get all digital event IDs in this recording
                    marker_ids = set(ea_i.labels)
                    for marker_id in marker_ids:
                        python_digievents = np.round(
                            ea_i.times.base[ea_i.labels == marker_id] * 30000).astype(int)
                        matlab_digievents = mts_ml[
                            np.nonzero(mid_ml == int(marker_id))]
                        assert_equal(python_digievents, matlab_digievents)

                        # Note: analog input events are not yet supported

    def test_segment_detection_reset(self):
        """
        This test makes sure segments are detected correctly when reset was used during recording.
        """

        # Path to nev that will fail
        filename_nev_fail = self.get_filename_path('segment/ResetFail/reset_fail')
        # Path to nsX and nev that will NOT fail
        filename = self.get_filename_path('segment/ResetCorrect/reset')

        # Warning filter needs to be set to always before first occurrence of this warning
        warnings.simplefilter("always", UserWarning)

        # This fails, because in the nev there is no way to separate two segments
        with self.assertRaises(AssertionError):
            reader = BlackrockIO(filename=filename, nsx_to_load=2, nev_override=filename_nev_fail)

        # The correct file will issue a warning because a reset has occurred
        # and could be detected, but was not explicitly documented in the file
        with warnings.catch_warnings(record=True) as w:
            reader = BlackrockIO(filename=filename, nsx_to_load=2)
            self.assertGreaterEqual(len(w), 1)
            messages = [str(warning.message) for warning in w if warning.category == UserWarning]
            self.assertIn("Detected 1 undocumented segments within nev data after "
                          "timestamps [5451].", messages)

        # Manually reset warning filter in order to not show too many warnings afterwards
        warnings.simplefilter("default")

        block = reader.read_block(load_waveforms=False, signal_group_mode="split-all")

        # 1 Segment at the beginning and 1 after reset
        self.assertEqual(len(block.segments), 2)
        # Checking all times are correct as read from file itself
        # (taking neo calculations into account)
        self.assertEqual(block.segments[0].t_start, 0.0)
        self.assertEqual(block.segments[0].t_stop, 4.02)
        # Clock is reset to 0
        self.assertEqual(block.segments[1].t_start, 0.0032)
        self.assertEqual(block.segments[1].t_stop, 3.9842)
        self.assertEqual(block.segments[0].analogsignals[0].t_start, 0.0)
        self.assertEqual(block.segments[0].analogsignals[0].t_stop, 4.02)
        self.assertEqual(block.segments[1].analogsignals[0].t_start, 0.0032)
        self.assertEqual(block.segments[1].analogsignals[0].t_stop, 3.9842)
        self.assertEqual(block.segments[0].spiketrains[0].t_start, 0.0)
        self.assertEqual(block.segments[0].spiketrains[0].t_stop, 4.02)
        self.assertEqual(block.segments[1].spiketrains[0].t_start, 0.0032)
        self.assertEqual(block.segments[1].spiketrains[0].t_stop, 3.9842)

        # Each segment must have the same number of analogsignals
        self.assertEqual(len(block.segments[0].analogsignals),
                         len(block.segments[1].analogsignals))

        # Length of analogsignals as created
        self.assertEqual(len(block.segments[0].analogsignals[0][:]), 4020)
        self.assertEqual(len(block.segments[1].analogsignals[0][:]), 3981)

    def test_segment_detection_pause(self):
        """
        This test makes sure segments are detected correctly when pause was used during recording.
        """

        # Path to nev that has spikes that don't fit nsX segment
        filename_nev_outside_seg = self.get_filename_path(
            'segment/PauseSpikesOutside/pause_spikes_outside_seg')
        # Path to nsX and nev that are correct
        filename = self.get_filename_path('segment/PauseCorrect/pause_correct')

        # This issues a warning, because there are spikes a long time after the last segment
        # And another one because there are spikes between segments
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reader = BlackrockIO(filename=filename, nsx_to_load=2,
                                 nev_override=filename_nev_outside_seg)
            self.assertGreaterEqual(len(w), 2)

            # Check that warnings are correct
            messages = [str(warning.message) for warning in w if warning.category == UserWarning]
            self.assertIn('Spikes outside any segment. Detected on segment #1', messages)
            self.assertIn('Spikes 0.0776s after last segment.', messages)

        block = reader.read_block(load_waveforms=False, signal_group_mode="split-all")

        # 2 segments
        self.assertEqual(len(block.segments), 2)

        # Checking all times are correct as read from file itself
        # (taking neo calculations into account)
        self.assertEqual(block.segments[0].t_start, 0.0)
        # This value is so high, because a spike occurred right before the second segment
        # And thus is added to the first segment
        # This is not normal behavior and occurs because of the way the files were cut
        # into test files
        self.assertAlmostEqual(block.segments[0].t_stop.magnitude, 15.83916667)
        # Clock is not reset
        self.assertEqual(block.segments[1].t_start.magnitude, 31.0087)
        # Segment time is longer here as well because of spikes after second segment
        self.assertEqual(block.segments[1].t_stop.magnitude, 35.0863)
        self.assertEqual(block.segments[0].analogsignals[0].t_start, 0.0)
        # The AnalogSignal is only 4 seconds long, as opposed to the segment
        # whose length is caused by the additional spike
        self.assertEqual(block.segments[0].analogsignals[0].t_stop, 4.0)
        self.assertEqual(block.segments[1].analogsignals[0].t_start, 31.0087)
        self.assertAlmostEqual(block.segments[1].analogsignals[0].t_stop.magnitude, 35.0087,
                               places=6)
        self.assertEqual(block.segments[0].spiketrains[0].t_start, 0.0)
        self.assertAlmostEqual(block.segments[0].spiketrains[0].t_stop.magnitude, 15.83916667,
                               places=8)
        self.assertEqual(block.segments[1].spiketrains[0].t_start, 31.0087)
        self.assertEqual(block.segments[1].spiketrains[0].t_stop, 35.0863)

        # Each segment has same number of analogsignals
        self.assertEqual(len(block.segments[0].analogsignals),
                         len(block.segments[1].analogsignals))

        # Analogsignals have exactly 4000 samples
        self.assertEqual(len(block.segments[0].analogsignals[0][:]), 4000)
        self.assertEqual(len(block.segments[1].analogsignals[0][:]), 4000)

        # This case is correct, no spikes outside segment or anything
        reader = BlackrockIO(filename=filename, nsx_to_load=2)
        block = reader.read_block(load_waveforms=False, signal_group_mode="split-all")

        # 2 segments
        self.assertEqual(len(block.segments), 2)

        # Checking all times are correct as read from file itself
        # (taking neo calculations into account)
        self.assertEqual(block.segments[0].t_start, 0.0)
        # Now segment time is only 4 seconds, because there were no additional spikes
        self.assertEqual(block.segments[0].t_stop, 4.0)
        self.assertEqual(block.segments[1].t_start, 31.0087)
        self.assertAlmostEqual(block.segments[1].t_stop.magnitude, 35.0087, places=6)
        self.assertEqual(block.segments[0].analogsignals[0].t_start, 0.0)
        self.assertEqual(block.segments[0].analogsignals[0].t_stop, 4.0)
        self.assertEqual(block.segments[1].analogsignals[0].t_start, 31.0087)
        self.assertAlmostEqual(block.segments[1].analogsignals[0].t_stop.magnitude, 35.0087,
                               places=6)
        self.assertEqual(block.segments[0].spiketrains[0].t_start, 0.0)
        self.assertEqual(block.segments[0].spiketrains[0].t_stop, 4.0)
        self.assertEqual(block.segments[1].spiketrains[0].t_start, 31.0087)
        self.assertAlmostEqual(block.segments[1].spiketrains[0].t_stop.magnitude, 35.0087,
                               places=6)

        # Each segment has same number of analogsignals
        self.assertEqual(len(block.segments[0].analogsignals),
                         len(block.segments[1].analogsignals))

        # ns2 was created in such a way that all analogsignals have 4000 samples
        self.assertEqual(len(block.segments[0].analogsignals[0][:]), 4000)
        self.assertEqual(len(block.segments[1].analogsignals[0][:]), 4000)


if __name__ == '__main__':
    unittest.main()
