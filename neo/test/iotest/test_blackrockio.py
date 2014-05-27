'''
Unit tests for neo.io.blackrockio.BlackrockIO
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io.blackrockio import BlackrockIO
from neo.test.iotest.common_io_test import BaseTestIO

import os.path
import tempfile
import numpy as np
import quantities as pq

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


class BlackrockIOTestCase(BaseTestIO, unittest.TestCase):

    files_to_test = ['FileSpec2.3001']

    files_to_download = ['FileSpec2.3001.nev',
                     'FileSpec2.3001.ns5',
                     'FileSpec2.3001.ccf',
                     'FileSpec2.3001.mat']
    ioclass = BlackrockIO


    def test_inputs_V23(self):
        """
        Test various inputs to BlackrockIO.read_block with version 2.3 file to check for parsing errors.
        """

        # Turns false on error
        allok = True

        try:
            b = BlackrockIO(os.path.join(tempfile.gettempdir(), 'files_for_testing_neo', 'blackrock', 'FileSpec2.3001'), print_diagnostic=False)

            # Load data to maximum extent, one None is not given as list
            block = b.read_block(n_starts=[None], n_stops=None, channel_list=range(1, 9), nsx=5, units=[], events=True, waveforms=False)
            lena = len(block.segments[0].analogsignals[0])
            numspa = len(block.segments[0].spiketrains[0])

            # Load data with very long extent using a negative time and the get_max_time() method
            block = b.read_block(n_starts=[-100 * pq.ms], n_stops=[b.get_max_time()], channel_list=range(1, 9), nsx=[5], units=[], events=False, waveforms=False)
            lenb = len(block.segments[0].analogsignals[0])
            numspb = len(block.segments[0].spiketrains[0])

            # Same length of analog signal? Both should have read the complete data set!
            if lena != lenb :
                allok = False
            # Same length of spike train? Both should have read the complete data set!
            if numspa != numspb:
                allok = False

            # Load data with very long extent, n_starts and n_stops not given as list
            block = b.read_block(n_starts=100 * b.nsx_unit[5], n_stops=200 * b.nsx_unit[5], channel_list=range(1, 9), nsx=5, units=[], events=False, waveforms=False)
            lena = len(block.segments[0].analogsignals[0])

            block = b.read_block(n_starts=301 * b.nsx_unit[5], n_stops=401 * b.nsx_unit[5], channel_list=range(1, 9), nsx=5, units=[], events=False, waveforms=False)
            lenb = len(block.segments[0].analogsignals[0])

            # Same length?
            if lena != lenb:
                allok = False

            # Length should be 100 samples exactly
            if lena != 100:
                allok = False

            # Load partial data types and check if this is selection is made
            block = b.read_block(n_starts=None, n_stops=None, channel_list=range(1, 9), nsx=5, units=None, events=False, waveforms=True)
            if len(block.segments) != 1:
                allok = False
            if len(block.segments[0].analogsignals) != 8:
                allok = False
            if len(block.recordingchannelgroups) != 8:
                allok = False
            if len(block.recordingchannelgroups[0].units) != 0:
                allok = False
            if len(block.segments[0].eventarrays) != 0:
                allok = False
            if len(block.segments[0].spiketrains) != 0:
                allok = False

            block = b.read_block(n_starts=[None, 3000 * pq.ms], n_stops=[1000 * pq.ms, None], channel_list=range(1, 9), nsx=None, units={1:0, 5:0, 6:0}, events=True, waveforms=True)
            if len(block.segments) != 2:
                allok = False
            if len(block.segments[0].analogsignals) != 0:
                allok = False
            if len(block.recordingchannelgroups) != 8:
                allok = False
            if len(block.recordingchannelgroups[0].units) != 1:
                allok = False
            # if len(block.recordingchannelgroups[4].units) != 0:  # only one of two neurons on channel 78, and only one unit for two segments!
            #    allok = False
            if len(block.segments[0].eventarrays) == 0:
                allok = False
            if len(block.segments[0].spiketrains[0].waveforms) == 0:
                allok = False

        except:
            allok = False

        self.assertTrue(allok)


    @unittest.skipUnless(HAVE_SCIPY, "requires scipy")
    def test_compare_blackrockio_with_matlabloader_V23(self):
        """
        This test compares the output of BlackrockIO.read_block() with the
        output generated by a Matlab implementation of a Blackrock file reader
        provided by the company. The output for comparison is provided in a .mat
        file created by the script create_data_matlab_blackrock.m.

        The function tests LFPs, spike times, and digital events on channels
        1-8 and spike waveforms on channel 1, unit 0.
        
        For details on the file contents, refer to FileSpec2.3.txt
        """

        # Turns false on error
        allok = True

        # Load data from Matlab generated files
        ml = scipy.io.loadmat(os.path.join(tempfile.gettempdir(), 'files_for_testing_neo', 'blackrock', 'FileSpec2.3001.mat'))
        lfp_ml = ml['lfp']  # (channel x time) LFP matrix
        ts_ml = ml['ts']  # spike time stamps
        elec_ml = ml['el']  # spike electrodes
        unit_ml = ml['un']  # spike unit IDs
        wf_ml = ml['wf']  # waveform unit 1 channel 1
        mts_ml = ml['mts']  # marker time stamps
        mid_ml = ml['mid']  # marker IDs

        # Load data in channels 1-3 from original data files using neo framework
        try:
            session = BlackrockIO(os.path.join(tempfile.gettempdir(), 'files_for_testing_neo', 'blackrock', 'FileSpec2.3001'),
                                        print_diagnostic=False)
            block = session.read_block(n_starts=[None], n_stops=[None],
                                       channel_list=range(1, 9), nsx=5, units=[],
                                       events=True, waveforms=True)
        except:
            allok = False

        # Check if analog data on channels 1-8 are equal
        for rcg_i in block.recordingchannelgroups:
            # Should only have one recording channel per group
            if len(rcg_i.recordingchannels) != 1:
                allok = False

            rc = rcg_i.recordingchannels[0]
            idx = rc.index
            if idx in range(1, 9):
                if np.any(rc.analogsignals[0].base - lfp_ml[idx - 1, :]):
                    allok = False

        # Should only have one segment
        if len(block.segments) != 1:
            allok = False

        # Check if spikes in channels 1,3,5,7 are equal
        for st_i in block.segments[0].spiketrains:
            channelid = st_i.annotations['channel_id']
            if channelid in range(1, 7, 2):
                unitid = st_i.annotations['unit_id']
                matlab_spikes = ts_ml[np.nonzero(np.logical_and(elec_ml == channelid, unit_ml == unitid))]
                if np.any(st_i.base - matlab_spikes):
                    allok = False

                # Check waveforms of channel 1, unit 0
                if channelid == 1 and unitid == 0:
                    if np.any(st_i.waveforms - wf_ml):
                        allok = False

        # Check if digital marker events are equal
        for ea_i in block.segments[0].eventarrays:
            if 'digital_marker' in ea_i.annotations.keys() and ea_i.annotations['digital_marker'] == True:
                markerid = ea_i.annotations['marker_id']
                matlab_digievents = mts_ml[np.nonzero(mid_ml == markerid)]
                if np.any(ea_i.times.base - matlab_digievents):
                    allok = False

        # Check if analog marker events are equal
        # Currently not implemented by the Matlab loader
        for ea_i in block.segments[0].eventarrays:
            if 'analog_marker' in ea_i.annotations.keys() and ea_i.annotations['analog_marker'] == True:
                markerid = ea_i.annotations['marker_id']
                matlab_anaevents = mts_ml[np.nonzero(mid_ml == markerid)]
                if np.any(ea_i.times.base - matlab_anaevents):
                    allok = False

        # Final result
        self.assertTrue(allok)


if __name__ == "__main__":
    unittest.main()
