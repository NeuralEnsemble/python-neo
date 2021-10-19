"""
Tests of neo.io.axographio
"""

from datetime import datetime
import unittest

from neo.io import AxographIO
from neo.test.iotest.common_io_test import BaseTestIO

import numpy as np
from numpy.testing import assert_equal
import quantities as pq
from neo.test.rawiotest.test_axographrawio import TestAxographRawIO

class TestAxographIO(BaseTestIO, unittest.TestCase):
    entities_to_download = [
        'axograph'
    ]
    entities_to_test = TestAxographRawIO.entities_to_test
    ioclass = AxographIO

    def test_version_1(self):
        """Test reading a version 1 AxoGraph file"""

        filename = self.get_local_path('axograph/AxoGraph_Graph_File')
        reader = AxographIO(filename=filename)
        blk = reader.read_block(signal_group_mode='split-all')
        assert_equal(blk.annotations['format_ver'], 1)

        names = [sig.name for sig in blk.segments[0].analogsignals]
        assert_equal(names, ['Current', 'Current'])

        sig = blk.segments[0].analogsignals[0][:5]
        arr = sig.as_array('pA')
        target = np.array([[-5.5078130],
                           [-3.1171880],
                           [+1.6640626],
                           [+1.6640626],
                           [+4.0546880]], dtype=np.float32)
        assert_equal(arr, target)

        assert_equal(sig.t_start, 0.0005000000237487257 * pq.s)

        assert_equal(sig.sampling_period, 0.0005000010132789612 * pq.s)

    def test_version_2(self):
        """Test reading a version 2 AxoGraph file"""

        filename = self.get_local_path('axograph/AxoGraph_Digitized_File')
        reader = AxographIO(filename=filename)
        blk = reader.read_block(signal_group_mode='split-all')
        assert_equal(blk.annotations['format_ver'], 2)

        names = [sig.name for sig in blk.segments[0].analogsignals]
        assert_equal(names, ['Current', 'Voltage', 'Column4', 'Column5',
                             'Column6', 'Column7', 'Column8', 'Column9',
                             'Column10', 'Column11', 'Column12', 'Column13',
                             'Column14', 'Column15', 'Column16', 'Column17',
                             'Column18', 'Column19', 'Column20', 'Column21',
                             'Column22', 'Column23', 'Column24', 'Column25',
                             'Column26', 'Column27', 'Column28', 'Column29'])

        sig = blk.segments[0].analogsignals[0][:5]
        arr = sig.as_array('pA')
        target = np.array([[0.3125],
                           [9.6875],
                           [9.6875],
                           [9.6875],
                           [9.3750]], dtype=np.float32)
        assert_equal(arr, target)

        assert_equal(sig.t_start, 0.00009999999747378752 * pq.s)

        assert_equal(sig.sampling_period, 0.00009999999747378750 * pq.s)

    def test_version_5(self):
        """Test reading a version 5 AxoGraph file"""

        filename = self.get_local_path('axograph/AxoGraph_X_File.axgx')
        reader = AxographIO(filename=filename)
        blk = reader.read_block(signal_group_mode='split-all')
        assert_equal(blk.annotations['format_ver'], 5)

        names = [sig.name for sig in blk.segments[0].analogsignals]
        assert_equal(names, ['Current', '', '', '', '', ''])

        sig = blk.segments[0].analogsignals[0][:5]
        arr = sig.as_array('pA')
        target = np.array([[+3.0846775],
                           [-2.5403225],
                           [-1.2903225],
                           [+6.8346770],
                           [-5.0403230]], dtype=np.float32)
        assert_equal(arr, target)

        assert_equal(sig.t_start, 0.00005 * pq.s)

        assert_equal(sig.sampling_period, 0.00005 * pq.s)

    def test_version_6(self):
        """Test reading a version 6 AxoGraph file"""

        filename = self.get_local_path('axograph/File_axograph.axgd')
        reader = AxographIO(filename=filename)
        blk = reader.read_block(signal_group_mode='split-all')
        assert_equal(blk.annotations['format_ver'], 6)

        names = [sig.name for sig in blk.segments[0].analogsignals]
        assert_equal(names, ['Membrane Voltage-1'])

        sig = blk.segments[0].analogsignals[0][:5]
        arr = sig.as_array('mV')
        target = np.array([[-60.731834],
                           [-60.701313],
                           [-60.670795],
                           [-60.701313],
                           [-60.731834]], dtype=np.float32)
        assert_equal(arr, target)

        assert_equal(sig.t_start, 0.00002 * pq.s)

        assert_equal(sig.sampling_period, 0.00002 * pq.s)

    def test_file_written_by_axographio_package_with_linearsequence(self):
        """Test reading file written by axographio package with linearsequence time column"""

        filename = self.get_local_path('axograph/written-by-axographio-with-linearsequence.axgx')
        reader = AxographIO(filename=filename)
        blk = reader.read_block(signal_group_mode='split-all')
        assert_equal(blk.annotations['format_ver'], 6)

        names = [sig.name for sig in blk.segments[0].analogsignals]
        assert_equal(names, ['Data 1', 'Data 2'])

        sig = blk.segments[0].analogsignals[0][:5]
        arr = sig.as_array('mV')
        target = np.array([[0.000000],
                           [9.999833],
                           [19.998667],
                           [29.995500],
                           [39.989334]], dtype=np.float32)
        assert_equal(arr, target)

        assert_equal(sig.t_start, 0 * pq.s)

        assert_equal(sig.sampling_period, 0.01 * pq.s)

    def test_file_written_by_axographio_package_without_linearsequence(self):
        """Test reading file written by axographio package without linearsequence time column"""

        filename = self.get_local_path('axograph/written-by-axographio-without-linearsequence.axgx')
        reader = AxographIO(filename=filename)
        blk = reader.read_block(signal_group_mode='split-all')
        assert_equal(blk.annotations['format_ver'], 6)

        names = [sig.name for sig in blk.segments[0].analogsignals]
        assert_equal(names, ['Data 1', 'Data 2'])

        sig = blk.segments[0].analogsignals[0][:5]
        arr = sig.as_array('mV')
        target = np.array([[0.000000],
                           [9.999833],
                           [19.998667],
                           [29.995500],
                           [39.989334]], dtype=np.float32)
        assert_equal(arr, target)

        assert_equal(sig.t_start, 0 * pq.s)

        assert_equal(sig.sampling_period, 0.009999999999999787 * pq.s)

    def test_file_with_corrupt_comment(self):
        """Test reading a file with a corrupt comment"""

        filename = self.get_local_path('axograph/corrupt-comment.axgx')
        reader = AxographIO(filename=filename)
        blk = reader.read_block(signal_group_mode='split-all')
        assert_equal(blk.annotations['format_ver'], 6)

        names = [sig.name for sig in blk.segments[0].analogsignals]
        assert_equal(names, ['Data 1', 'Data 2'])

        sig = blk.segments[0].analogsignals[0][:5]
        arr = sig.as_array('mV')
        target = np.array([[0.000000],
                           [9.999833],
                           [19.998667],
                           [29.995500],
                           [39.989334]], dtype=np.float32)
        assert_equal(arr, target)

        assert_equal(sig.t_start, 0 * pq.s)

        assert_equal(sig.sampling_period, 0.01 * pq.s)

    def test_multi_segment(self):
        """Test reading an episodic file into multiple Segments"""

        filename = self.get_local_path('axograph/episodic.axgd')
        reader = AxographIO(filename=filename)
        blk = reader.read_block(signal_group_mode='split-all')

        assert_equal(len(blk.segments), 30)
        assert_equal(len(blk.groups), 2)
        assert_equal(len(blk.segments[0].analogsignals), 2)

        names = [sig.name for sig in blk.segments[0].analogsignals]
        assert_equal(names, ['CAP', 'STIM'])

        sig = blk.segments[0].analogsignals[0][:5]
        arr = sig.as_array('V')
        target = np.array([[1.37500e-06],
                           [1.53125e-06],
                           [1.34375e-06],
                           [1.09375e-06],
                           [1.21875e-06]], dtype=np.float32)
        assert_equal(arr, target)

    def test_force_single_segment(self):
        """Test reading an episodic file into one Segment"""

        filename = self.get_local_path('axograph/episodic.axgd')
        reader = AxographIO(filename=filename, force_single_segment=True)
        blk = reader.read_block(signal_group_mode='split-all')

        assert_equal(len(blk.segments), 1)
        assert_equal(len(blk.groups), 60)
        assert_equal(len(blk.segments[0].analogsignals), 60)

        names = [sig.name for sig in blk.segments[0].analogsignals]
        assert_equal(names, ['CAP', 'STIM'] * 30)

        sig = blk.segments[0].analogsignals[0][:5]
        arr = sig.as_array('V')
        target = np.array([[1.37500e-06],
                           [1.53125e-06],
                           [1.34375e-06],
                           [1.09375e-06],
                           [1.21875e-06]], dtype=np.float32)
        assert_equal(arr, target)

    def test_group_by_same_units(self):
        """Test reading with group-by-same-units"""

        filename = self.get_local_path('axograph/episodic.axgd')
        reader = AxographIO(filename=filename)
        blk = reader.read_block(signal_group_mode='group-by-same-units')

        assert_equal(len(blk.segments), 30)
        assert_equal(len(blk.groups), 1)
        assert_equal(len(blk.segments[0].analogsignals), 1)

        chan_names = blk.segments[0].analogsignals[0].array_annotations['channel_names']
        assert_equal(chan_names, ['CAP', 'STIM'])

        sig = blk.segments[0].analogsignals[0][:5]
        arr = sig.as_array('V')
        target = np.array([[1.37500e-06, 3.43750e-03],
                           [1.53125e-06, 2.81250e-03],
                           [1.34375e-06, 1.87500e-03],
                           [1.09375e-06, 1.56250e-03],
                           [1.21875e-06, 1.56250e-03]], dtype=np.float32)
        assert_equal(arr, target)

    def test_events_and_epochs(self):
        """Test loading events and epochs"""

        filename = self.get_local_path('axograph/events_and_epochs.axgx')
        reader = AxographIO(filename=filename)
        blk = reader.read_block(signal_group_mode='split-all')

        event = blk.segments[0].events[0]
        assert_equal(event.times, [5999, 5999, 23499, 23499,
                                   26499, 26499, 35999]
                     * blk.segments[0].analogsignals[0].sampling_period)
        assert_equal(event.labels, ['Stop', 'Start', 'Stop', 'Start',
                                    'Stop', 'Start', 'Stop'])

        epoch = blk.segments[0].epochs[0]
        assert_equal(epoch.times, np.array([0.1, 4]) * pq.s)
        assert_equal(epoch.durations, np.array([1.4, 2]) * pq.s)
        assert_equal(epoch.labels, ['test interval 1', 'test interval 2'])

    def test_rec_datetime(self):
        """Test parsing the recording datetime from notes"""

        # parsing of rec_datetime differs depending on acquisition mode

        # file obtained in episodic acquisition mode has date and time on
        # separate lines of notes
        filename = self.get_local_path('axograph/episodic.axgd')
        reader = AxographIO(filename=filename)
        blk = reader.read_block(signal_group_mode='split-all')
        assert_equal(blk.rec_datetime, datetime(2018, 6, 7, 15, 11, 36))

        # file obtained in continuous acquisition mode has date and time in
        # single line of notes
        filename = self.get_local_path('axograph/events_and_epochs.axgx')
        reader = AxographIO(filename=filename)
        blk = reader.read_block(signal_group_mode='split-all')
        assert_equal(blk.rec_datetime, datetime(2019, 5, 25, 20, 16, 25))


if __name__ == "__main__":
    unittest.main()
