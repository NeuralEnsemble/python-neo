# -*- coding: utf-8 -*-
"""
Tests of neo.io.blackrockio
"""

# needed for python 3 compatibility
from __future__ import absolute_import

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
        'FileSpec2.3001.mat']

    ioclass = BlackrockIO
    
    def test_load_waveforms(self):
        reader = BlackrockIO(filename=
                get_test_file_full_path(
                    ioclass=BlackrockIO,
                    filename='FileSpec2.3001',
                    directory=self.local_test_dir, clean=False),
                verbose=False)
        
        bl = reader.read_block(load_waveforms=True)
        assert_neo_object_is_compliant(bl)
        
        
    #API of BlackrockIO V4 had special adaptation
    #with standart neo.io API like n_starts, n_stops that create
    #fake segment clip by that time limits
    #all that test will fail because of that
    
    #~ def test_inputs_V23(self):
        #~ """
        #~ Test various inputs to BlackrockIO.read_block with version 2.3 file
        #~ to check for parsing errors.
        #~ """

        #~ try:
            #~ b = BlackrockIO(
                #~ get_test_file_full_path(
                    #~ ioclass=BlackrockIO,
                    #~ filename='FileSpec2.3001',
                    #~ directory=self.local_test_dir, clean=False),
                #~ verbose=False)
        #~ except:
            #~ self.fail()

        #~ # Load data to maximum extent, one None is not given as list
        #~ block = b.read_block(
            #~ n_starts=[None], n_stops=None, channels=range(1, 9),
            #~ nsx_to_load=5, units='all', load_events=True,
            #~ load_waveforms=False)
        #~ lena = len(block.segments[0].analogsignals[0])
        #~ numspa = len(block.segments[0].spiketrains[0])

        #~ # Load data using a negative time and a time exceeding the end of the
        #~ # recording
        #~ too_large_tstop = block.segments[0].analogsignals[0].t_stop + 1 * pq.s
        #~ block = b.read_block(
            #~ n_starts=[-100 * pq.ms], n_stops=[too_large_tstop],
            #~ channels=range(1, 9), nsx_to_load=[5], units='all',
            #~ load_events=False, load_waveforms=False)
        #~ lenb = len(block.segments[0].analogsignals[0])
        #~ numspb = len(block.segments[0].spiketrains[0])

        #~ # Same length of analog signal?
        #~ # Both should have read the complete data set!
        #~ self.assertEqual(lena, lenb)

        #~ # Same length of spike train?
        #~ # Both should have read the complete data set!
        #~ self.assertEqual(numspa, numspb)

        #~ # n_starts and n_stops not given as list
        #~ # verifies identical length of returned signals given equal durations
        #~ # as input
        #~ ns5_unit = block.segments[0].analogsignals[0].sampling_period
        #~ block = b.read_block(
            #~ n_starts=100 * ns5_unit, n_stops=200 * ns5_unit,
            #~ channels=range(1, 9), nsx_to_load=5, units='all',
            #~ load_events=False, load_waveforms=False)
        #~ lena = len(block.segments[0].analogsignals[0])

        #~ block = b.read_block(
            #~ n_starts=301 * ns5_unit, n_stops=401 * ns5_unit,
            #~ channels=range(1, 9), nsx_to_load=5, units='all',
            #~ load_events=False, load_waveforms=False)
        #~ lenb = len(block.segments[0].analogsignals[0])

        #~ # Same length?
        #~ self.assertEqual(lena, lenb)
        #~ # Length should be 100 samples exactly
        #~ self.assertEqual(lena, 100)

        #~ # Load partial data types and check if this is selection is made
        #~ block = b.read_block(
            #~ n_starts=None, n_stops=None, channels=range(1, 9),
            #~ nsx_to_load=5, units='none', load_events=False,
            #~ load_waveforms=True)

        #~ self.assertEqual(len(block.segments), 1)
        #~ self.assertEqual(len(block.segments[0].analogsignals), 8)
        #~ self.assertEqual(len(block.channel_indexes), 8)
        #~ self.assertEqual(len(block.channel_indexes[0].units), 0)
        #~ self.assertEqual(len(block.segments[0].events), 0)
        #~ self.assertEqual(len(block.segments[0].spiketrains), 0)

        #~ # NOTE: channel 6 does not contain any unit
        #~ block = b.read_block(
            #~ n_starts=[None, 3000 * pq.ms], n_stops=[1000 * pq.ms, None],
            #~ channels=range(1, 9), nsx_to_load='none',
            #~ units={1: 0, 5: 0, 6: 0}, load_events=True,
            #~ load_waveforms=True)

        #~ self.assertEqual(len(block.segments), 2)
        #~ self.assertEqual(len(block.segments[0].analogsignals), 0)
        #~ self.assertEqual(len(block.channel_indexes), 8)
        #~ self.assertEqual(len(block.channel_indexes[0].units), 1)
        #~ self.assertEqual(len(block.segments[0].events), 0)
        #~ self.assertEqual(len(block.segments[0].spiketrains), 2)

if __name__ == '__main__':
    unittest.main()
