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
        'FileSpec2.3001.mat']

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
        reader = BlackrockIO(filename=filename, verbose=False, nsx_to_load=5,)
        
        
        # Load data to maximum extent, one None is not given as list
        block = reader.read_block(time_slices=None,  load_waveforms=False)
        lena = len(block.segments[0].analogsignals[0])
        numspa = len(block.segments[0].spiketrains[0])

        # Load data using a negative time and a time exceeding the end of the
        # recording raise an error
        too_large_tstop = block.segments[0].analogsignals[0].t_stop + 1 * pq.s
        buggy_slice = (-100 * pq.ms, too_large_tstop)

        #this raise error in read_block
        with self.assertRaises(ValueError):
            block = reader.read_block(time_slices=[buggy_slice])
        
        #but this is valid in read_segment because seg_index is specified
        seg = reader.read_segment(seg_index=0, time_slice=buggy_slice)
        
        lenb = len(seg.analogsignals[0])
        numspb = len(seg.spiketrains[0])

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
        time_slice = (100 * ns5_unit, 200 * ns5_unit)
        block = reader.read_block(time_slices=[time_slice])
        lena = len(block.segments[0].analogsignals[0])
        
        time_slice = (100 * ns5_unit, 200 * ns5_unit)
        block = reader.read_block(time_slices=[time_slice])
        lenb = len(block.segments[0].analogsignals[0])

        # Same length?
        self.assertEqual(lena, lenb)
        # Length should be 100 samples exactly
        self.assertEqual(lena, 100)

        # test 4 Units
        time_slices=[(0, 1000*pq.ms), (3000*pq.ms, 4000*pq.ms)]
        block = reader.read_block(time_slices=time_slices, load_waveforms=True,
                    units_group_mode='all-in-one')

        self.assertEqual(len(block.segments), 2)
        self.assertEqual(len(block.segments[0].analogsignals), 10)
        self.assertEqual(len(block.channel_indexes[-1].units), 4)
        self.assertEqual(len(block.channel_indexes[-1].units), 
                    len(block.segments[0].spiketrains))
        
        anasig = block.segments[0].analogsignals[0]
        self.assertIsNotNone(anasig.file_origin)
        


if __name__ == '__main__':
    unittest.main()
