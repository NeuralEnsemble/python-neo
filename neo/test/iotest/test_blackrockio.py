# -*- coding: utf-8 -*-
"""
Tests of neo.io.blackrockio
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import unittest

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

    files_to_test = ['FileSpec2.3001']

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
        'blackrock_2_1/l101210-001-02.nev']

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
        block = reader.read_block(load_waveforms=False)
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
                                  units_group_mode='all-in-one')

        self.assertEqual(len(block.segments[0].analogsignals), 96)
        self.assertEqual(len(block.channel_indexes[-1].units), 218)
        self.assertEqual(len(block.channel_indexes[-1].units),
                         len(block.segments[0].spiketrains))

        anasig = block.segments[0].analogsignals[0]
        self.assertIsNotNone(anasig.file_origin)

    @unittest.skipUnless(HAVE_SCIPY, "requires scipy")
    def test_compare_blackrockio_with_matlabloader_v21(self):
        """
        This test compares the output of ReachGraspIO.read_block() with the
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
            block = session.read_block(load_waveforms=True)
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


if __name__ == '__main__':
    unittest.main()
