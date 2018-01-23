# -*- coding: utf-8 -*-
"""
Tests of neo.io.exampleio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division
import warnings

import unittest

import quantities as pq
import numpy as np

from neo.io.nestio import ColumnIO
from neo.io.nestio import NestIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.iotest.tools import get_test_file_full_path


class TestNestIO_Analogsignals(BaseTestIO, unittest.TestCase):
    ioclass = NestIO
    files_to_test = []
    files_to_download = ['0gid-1time-2gex-3Vm-1261-0.dat',
                         '0gid-1time-2gex-1262-0.dat',
                         '0gid-1time-2Vm-3gex-4gin-1260-0.dat',
                         '0gid-1time-2Vm-3Iex-4Iin-1264-0.dat',
                         '0gid-1time-2Vm-1259-0.dat',
                         '0gid-1time-1256-0.gdf',
                         '0gid-1time_in_steps-2Vm-1263-0.dat',
                         '0gid-1time_in_steps-1258-0.gdf',
                         '0time-1255-0.gdf',
                         '0time_in_steps-1257-0.gdf',
                         'brunel-delta-nest_mod.py',
                         'N1-0gid-1time-2Vm-1265-0.dat',
                         'N1-0time-1Vm-1266-0.dat',
                         'N1-0Vm-1267-0.dat']

    def test_read_analogsignal(self):
        """
        Tests reading files in the 2 different formats:
        - with GIDs, with times as floats
        - with GIDs, with time as integer
        """

        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-2gex-3Vm-1261-0.dat',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        r.read_analogsignal(gid=1, t_stop=1000. * pq.ms,
                            sampling_period=pq.ms, lazy=False,
                            id_column=0, time_column=1,
                            value_column=2, value_type='V_m')
        r.read_segment(gid_list=[1], t_stop=1000. * pq.ms,
                       sampling_period=pq.ms, lazy=False, id_column_dat=0,
                       time_column_dat=1, value_columns_dat=2,
                       value_types='V_m')

        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time_in_steps-2Vm-1263-0.dat',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        r.read_analogsignal(gid=1, t_stop=1000. * pq.ms,
                            time_unit=pq.CompoundUnit('0.1*ms'),
                            sampling_period=pq.ms, lazy=False,
                            id_column=0, time_column=1,
                            value_column=2, value_type='V_m')
        r.read_segment(gid_list=[1], t_stop=1000. * pq.ms,
                       time_unit=pq.CompoundUnit('0.1*ms'),
                       sampling_period=pq.ms, lazy=False, id_column_dat=0,
                       time_column_dat=1, value_columns_dat=2,
                       value_types='V_m')

        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-2Vm-1259-0.dat',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        r.read_analogsignal(gid=1, t_stop=1000. * pq.ms,
                            time_unit=pq.CompoundUnit('0.1*ms'),
                            sampling_period=pq.ms, lazy=False,
                            id_column=0, time_column=1,
                            value_column=2, value_type='V_m')
        r.read_segment(gid_list=[1], t_stop=1000. * pq.ms,
                       time_unit=pq.CompoundUnit('0.1*ms'),
                       sampling_period=pq.ms, lazy=False, id_column_dat=0,
                       time_column_dat=1, value_columns_dat=2,
                       value_types='V_m')

    def test_id_column_none_multiple_neurons(self):
        """
        Tests if function correctly raises an error if the user tries to read
        from a file which does not contain unit IDs, but data for multiple
        units.
        """

        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0time-1255-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        with self.assertRaises(ValueError):
            r.read_analogsignal(t_stop=1000. * pq.ms, lazy=False,
                                sampling_period=pq.ms,
                                id_column=None, time_column=0,
                                value_column=1)
            r.read_segment(t_stop=1000. * pq.ms, lazy=False,
                           sampling_period=pq.ms, id_column_gdf=None,
                           time_column_gdf=0)

    def test_values(self):
        """
        Tests if the function returns the correct values.
        """

        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-2gex-3Vm-1261-0.dat',
                directory=self.local_test_dir, clean=False)

        id_to_test = 1
        r = NestIO(filenames=filename)
        seg = r.read_segment(gid_list=[id_to_test],
                             t_stop=1000. * pq.ms,
                             sampling_period=pq.ms, lazy=False,
                             id_column_dat=0, time_column_dat=1,
                             value_columns_dat=2, value_types='V_m')

        dat = np.loadtxt(filename)
        target_data = dat[:, 2][np.where(dat[:, 0] == id_to_test)]
        target_data = target_data[:, None]
        st = seg.analogsignals[0]
        np.testing.assert_array_equal(st.magnitude, target_data)

    def test_read_segment(self):
        """
        Tests if signals are correctly stored in a segment.
        """

        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-2gex-1262-0.dat',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)

        id_list_to_test = range(1, 10)
        seg = r.read_segment(gid_list=id_list_to_test,
                             t_stop=1000. * pq.ms,
                             sampling_period=pq.ms, lazy=False,
                             id_column_dat=0, time_column_dat=1,
                             value_columns_dat=2, value_types='V_m')

        self.assertTrue(len(seg.analogsignals) == len(id_list_to_test))

        id_list_to_test = []
        seg = r.read_segment(gid_list=id_list_to_test,
                             t_stop=1000. * pq.ms,
                             sampling_period=pq.ms, lazy=False,
                             id_column_dat=0, time_column_dat=1,
                             value_columns_dat=2, value_types='V_m')

        self.assertEqual(len(seg.analogsignals), 50)

    def test_read_block(self):
        """
        Tests if signals are correctly stored in a block.
        """

        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-2gex-1262-0.dat',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)

        id_list_to_test = range(1, 10)
        blk = r.read_block(gid_list=id_list_to_test,
                          t_stop=1000. * pq.ms,
                          sampling_period=pq.ms, lazy=False,
                          id_column_dat=0, time_column_dat=1,
                          value_columns_dat=2, value_types='V_m')

        self.assertTrue(len(blk.segments[0].analogsignals) == len(id_list_to_test))

    def test_wrong_input(self):
        """
        Tests two cases of wrong user input, namely
        - User does not specify a value column
        - User does not make any specifications
        - User does not define sampling_period as a unit
        - User specifies a non-default value type without
          specifying a value_unit
        - User specifies t_start < 1.*sampling_period
        """

        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-2gex-1262-0.dat',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        with self.assertRaises(ValueError):
            r.read_segment(t_stop=1000. * pq.ms, lazy=False,
                           id_column_dat=0, time_column_dat=1)
        with self.assertRaises(ValueError):
            r.read_segment()
        with self.assertRaises(ValueError):
            r.read_segment(gid_list=[1], t_stop=1000. * pq.ms,
                           sampling_period=1. * pq.ms, lazy=False,
                           id_column_dat=0, time_column_dat=1,
                           value_columns_dat=2, value_types='V_m')

        with self.assertRaises(ValueError):
            r.read_segment(gid_list=[1], t_stop=1000. * pq.ms,
                           sampling_period=pq.ms, lazy=False,
                           id_column_dat=0, time_column_dat=1,
                           value_columns_dat=2, value_types='U_mem')

    def test_t_start_t_stop(self):
        """
        Test for correct t_start and t_stop values of AnalogSignalArrays.
        """
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-2gex-1262-0.dat',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)

        t_start_targ = 450. * pq.ms
        t_stop_targ = 480. * pq.ms

        seg = r.read_segment(gid_list=[], t_start=t_start_targ,
                             t_stop=t_stop_targ, lazy=False,
                             id_column_dat=0, time_column_dat=1,
                             value_columns_dat=2, value_types='V_m')
        anasigs = seg.analogsignals
        for anasig in anasigs:
            self.assertTrue(anasig.t_start == t_start_targ)
            self.assertTrue(anasig.t_stop == t_stop_targ)

    def test_notimeid(self):
        """
        Test for warning, when no time column id was provided.
        """

        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-2gex-1262-0.dat',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)

        t_start_targ = 450. * pq.ms
        t_stop_targ = 460. * pq.ms
        sampling_period = pq.CompoundUnit('5*ms')

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            seg = r.read_segment(gid_list=[], t_start=t_start_targ,
                                 sampling_period=sampling_period,
                                 t_stop=t_stop_targ, lazy=False,
                                 id_column_dat=0, time_column_dat=None,
                                 value_columns_dat=2, value_types='V_m')
            # Verify number and content of warning
            self.assertEqual(len(w), 1)
            self.assertIn("no time column id", str(w[0].message))
        sts = seg.analogsignals
        for st in sts:
            self.assertTrue(st.t_start == 1 * 5 * pq.ms)
            self.assertTrue(
                    st.t_stop == len(st) * sampling_period + 1 * 5 * pq.ms)

    def test_multiple_value_columns(self):
        """
        Test for simultaneous loading of multiple columns from dat file.
        """

        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-2Vm-3Iex-4Iin-1264-0.dat',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)

        sampling_period = pq.CompoundUnit('5*ms')
        seg = r.read_segment(gid_list=[1001],
                             value_columns_dat=[2, 3],
                             sampling_period=sampling_period)
        anasigs = seg.analogsignals
        self.assertEqual(len(anasigs), 2)

    def test_single_gid(self):
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='N1-0gid-1time-2Vm-1265-0.dat',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        anasig = r.read_analogsignal(gid=1, t_stop=1000. * pq.ms,
                                     time_unit=pq.CompoundUnit('0.1*ms'),
                                     sampling_period=pq.ms, lazy=False,
                                     id_column=0, time_column=1,
                                     value_column=2, value_type='V_m')
        assert anasig.annotations['id'] == 1

    def test_no_gid(self):
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='N1-0time-1Vm-1266-0.dat',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        anasig = r.read_analogsignal(gid=None, t_stop=1000. * pq.ms,
                                     time_unit=pq.CompoundUnit('0.1*ms'),
                                     sampling_period=pq.ms, lazy=False,
                                     id_column=None, time_column=0,
                                     value_column=1, value_type='V_m')
        self.assertEqual(anasig.annotations['id'], None)
        self.assertEqual(len(anasig), 19)

    def test_no_gid_no_time(self):
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='N1-0Vm-1267-0.dat',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        anasig = r.read_analogsignal(gid=None,
                                     sampling_period=pq.ms, lazy=False,
                                     id_column=None, time_column=None,
                                     value_column=0, value_type='V_m')
        self.assertEqual(anasig.annotations['id'], None)
        self.assertEqual(len(anasig), 19)


class TestNestIO_Spiketrains(BaseTestIO, unittest.TestCase):
    ioclass = NestIO
    files_to_test = []
    files_to_download = []

    def test_read_spiketrain(self):
        """
        Tests reading files in the 4 different formats:
        - without GIDs, with times as floats
        - without GIDs, with times as integers in time steps
        - with GIDs, with times as floats
        - with GIDs, with times as integers in time steps
        """
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0time-1255-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        r.read_spiketrain(t_start=400. * pq.ms, t_stop=500. * pq.ms, lazy=False,
                          id_column=None, time_column=0)
        r.read_segment(t_start=400. * pq.ms, t_stop=500. * pq.ms, lazy=False,
                       id_column_gdf=None, time_column_gdf=0)

        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0time_in_steps-1257-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        r.read_spiketrain(t_start=400. * pq.ms, t_stop=500. * pq.ms,
                          time_unit=pq.CompoundUnit('0.1*ms'), lazy=False,
                          id_column=None, time_column=0)
        r.read_segment(t_start=400. * pq.ms, t_stop=500. * pq.ms,
                       time_unit=pq.CompoundUnit('0.1*ms'), lazy=False,
                       id_column_gdf=None, time_column_gdf=0)

        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-1256-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        r.read_spiketrain(gdf_id=1, t_start=400. * pq.ms, t_stop=500. * pq.ms,
                          lazy=False, id_column_gdf=0, time_column_gdf=1)
        r.read_segment(gid_list=[1], t_start=400. * pq.ms, t_stop=500. * pq.ms,
                       lazy=False, id_column_gdf=0, time_column_gdf=1)

        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time_in_steps-1258-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        r.read_spiketrain(gdf_id=1, t_start=400. * pq.ms, t_stop=500. * pq.ms,
                          time_unit=pq.CompoundUnit('0.1*ms'), lazy=False,
                          id_column=0, time_column=1)
        r.read_segment(gid_list=[1], t_start=400. * pq.ms, t_stop=500. * pq.ms,
                       time_unit=pq.CompoundUnit('0.1*ms'), lazy=False,
                       id_column_gdf=0, time_column_gdf=1)

    def test_read_integer(self):
        """
        Tests if spike times are actually stored as integers if they are stored
        in time steps in the file.
        """
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0time_in_steps-1257-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        st = r.read_spiketrain(gdf_id=None, t_start=400. * pq.ms,
                               t_stop=500. * pq.ms,
                               time_unit=pq.CompoundUnit('0.1*ms'),
                               lazy=False, id_column=None, time_column=0)
        self.assertTrue(st.magnitude.dtype == np.int32)
        seg = r.read_segment(gid_list=[None], t_start=400. * pq.ms,
                             t_stop=500. * pq.ms,
                             time_unit=pq.CompoundUnit('0.1*ms'),
                             lazy=False, id_column_gdf=None, time_column_gdf=0)
        sts = seg.spiketrains
        self.assertTrue(all([st.magnitude.dtype == np.int32 for st in sts]))

        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time_in_steps-1258-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(
                filenames=filename)
        st = r.read_spiketrain(gdf_id=1, t_start=400. * pq.ms,
                               t_stop=500. * pq.ms,
                               time_unit=pq.CompoundUnit('0.1*ms'),
                               lazy=False, id_column=0, time_column=1)
        self.assertTrue(st.magnitude.dtype == np.int32)
        seg = r.read_segment(gid_list=[1], t_start=400. * pq.ms,
                             t_stop=500. * pq.ms,
                             time_unit=pq.CompoundUnit('0.1*ms'),
                             lazy=False, id_column_gdf=0, time_column_gdf=1)
        sts = seg.spiketrains
        self.assertTrue(all([st.magnitude.dtype == np.int32 for st in sts]))

    def test_read_float(self):
        """
        Tests if spike times are stored as floats if they
        are stored as floats in the file.
        """
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-1256-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        st = r.read_spiketrain(gdf_id=1, t_start=400. * pq.ms,
                               t_stop=500. * pq.ms,
                               lazy=False, id_column=0, time_column=1)
        self.assertTrue(st.magnitude.dtype == np.float)
        seg = r.read_segment(gid_list=[1], t_start=400. * pq.ms,
                             t_stop=500. * pq.ms,
                             lazy=False, id_column_gdf=0, time_column_gdf=1)
        sts = seg.spiketrains
        self.assertTrue(all([s.magnitude.dtype == np.float for s in sts]))

    def test_values(self):
        """
        Tests if the routine loads the correct numbers from the file.
        """
        id_to_test = 1
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-1256-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        seg = r.read_segment(gid_list=[id_to_test],
                             t_start=400. * pq.ms,
                             t_stop=500. * pq.ms, lazy=False,
                             id_column_gdf=0, time_column_gdf=1)

        dat = np.loadtxt(filename)
        target_data = dat[:, 1][np.where(dat[:, 0] == id_to_test)]

        st = seg.spiketrains[0]
        np.testing.assert_array_equal(st.magnitude, target_data)

    def test_read_segment(self):
        """
        Tests if spiketrains are correctly stored in a segment.
        """
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-1256-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)

        id_list_to_test = range(1, 10)
        seg = r.read_segment(gid_list=id_list_to_test, t_start=400. * pq.ms,
                             t_stop=500. * pq.ms, lazy=False,
                             id_column_gdf=0, time_column_gdf=1)
        self.assertTrue(len(seg.spiketrains) == len(id_list_to_test))

        id_list_to_test = []
        seg = r.read_segment(gid_list=id_list_to_test, t_start=400. * pq.ms,
                             t_stop=500. * pq.ms, lazy=False,
                             id_column_gdf=0, time_column_gdf=1)
        self.assertTrue(len(seg.spiketrains) == 50)

    def test_read_segment_accepts_range(self):
        """
        Tests if spiketrains can be retrieved by specifying a range of GDF IDs.
        """
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-1256-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)

        seg = r.read_segment(gid_list=(10, 39), t_start=400. * pq.ms,
                             t_stop=500. * pq.ms, lazy=False,
                             id_column_gdf=0, time_column_gdf=1)
        self.assertEqual(len(seg.spiketrains), 30)

    def test_read_segment_range_is_reasonable(self):
        """
        Tests if error is thrown correctly, when second entry is smaller than
        the first one of the range.
        """
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-1256-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)

        seg = r.read_segment(gid_list=(10, 10), t_start=400. * pq.ms,
                             t_stop=500. * pq.ms, lazy=False,
                             id_column_gdf=0, time_column_gdf=1)
        self.assertEqual(len(seg.spiketrains), 1)
        with self.assertRaises(ValueError):
            r.read_segment(gid_list=(10, 9), t_start=400. * pq.ms,
                           t_stop=500. * pq.ms, lazy=False,
                           id_column_gdf=0, time_column_gdf=1)

    def test_read_spiketrain_annotates(self):
        """
        Tests if correct annotation is added when reading a spike train.
        """
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-1256-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        ID = 7
        st = r.read_spiketrain(gdf_id=ID, t_start=400. * pq.ms,
                               t_stop=500. * pq.ms)
        self.assertEqual(ID, st.annotations['id'])

    def test_read_segment_annotates(self):
        """
        Tests if correct annotation is added when reading a segment.
        """
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-1256-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        IDs = (5, 11)
        sts = r.read_segment(gid_list=(5, 11), t_start=400. * pq.ms,
                             t_stop=500. * pq.ms)
        for ID in np.arange(5, 12):
            self.assertEqual(ID, sts.spiketrains[ID - 5].annotations['id'])

    def test_adding_custom_annotation(self):
        """
        Tests if custom annotation is correctly added.
        """
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-1256-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        st = r.read_spiketrain(gdf_id=0, t_start=400. * pq.ms,
                               t_stop=500. * pq.ms,
                               layer='L23', population='I')
        self.assertEqual(0, st.annotations.pop('id'))
        self.assertEqual('L23', st.annotations.pop('layer'))
        self.assertEqual('I', st.annotations.pop('population'))
        self.assertEqual({}, st.annotations)

    def test_wrong_input(self):
        """
        Tests two cases of wrong user input, namely
        - User does not specify neuron IDs although the file contains IDs.
        - User does not make any specifications.
        """
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-1256-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        with self.assertRaises(ValueError):
            r.read_segment(t_start=400. * pq.ms, t_stop=500. * pq.ms,
                           lazy=False,
                           id_column_gdf=0, time_column_gdf=1)
        with self.assertRaises(ValueError):
            r.read_segment()

    def test_t_start_t_stop(self):
        """
        Tests if the t_start and t_stop arguments are correctly processed.
        """
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-1256-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)

        t_stop_targ = 490. * pq.ms
        t_start_targ = 410. * pq.ms

        seg = r.read_segment(gid_list=[], t_start=t_start_targ,
                             t_stop=t_stop_targ, lazy=False,
                             id_column_gdf=0, time_column_gdf=1)
        sts = seg.spiketrains
        self.assertTrue(np.max([np.max(st.magnitude) for st in sts
                                if len(st) > 0])
                        < t_stop_targ.rescale(sts[0].times.units).magnitude)
        self.assertTrue(np.min([np.min(st.magnitude) for st in sts
                                if len(st) > 0])
                        >= t_start_targ.rescale(sts[0].times.units).magnitude)

    def test_t_start_undefined_raises_error(self):
        """
        Tests if undefined t_start, i.e., t_start=None raises error.
        """
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-1256-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        with self.assertRaises(ValueError):
            r.read_spiketrain(gdf_id=1, t_stop=500. * pq.ms, lazy=False,
                              id_column=0, time_column=1)
        with self.assertRaises(ValueError):
            r.read_segment(gid_list=[1, 2, 3], t_stop=500. * pq.ms, lazy=False,
                           id_column_gdf=0, time_column_gdf=1)

    def test_t_stop_undefined_raises_error(self):
        """
        Tests if undefined t_stop, i.e., t_stop=None raises error.
        """
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-1256-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        with self.assertRaises(ValueError):
            r.read_spiketrain(gdf_id=1, t_start=400. * pq.ms, lazy=False,
                              id_column=0, time_column=1)
        with self.assertRaises(ValueError):
            r.read_segment(gid_list=[1, 2, 3], t_start=400. * pq.ms, lazy=False,
                           id_column_gdf=0, time_column_gdf=1)

    def test_gdf_id_illdefined_raises_error(self):
        """
        Tests if ill-defined gdf_id in read_spiketrain (i.e., None, list, or
        empty list) raises error.
        """
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-1256-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        with self.assertRaises(ValueError):
            r.read_spiketrain(gdf_id=[], t_start=400. * pq.ms,
                              t_stop=500. * pq.ms)
        with self.assertRaises(ValueError):
            r.read_spiketrain(gdf_id=[1], t_start=400. * pq.ms,
                              t_stop=500. * pq.ms)
        with self.assertRaises(ValueError):
            r.read_spiketrain(t_start=400. * pq.ms, t_stop=500. * pq.ms)

    def test_read_segment_can_return_empty_spiketrains(self):
        """
        Tests if read_segment makes sure that only non-zero spike trains are
        returned.
        """
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-1256-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        seg = r.read_segment(gid_list=[], t_start=400. * pq.ms,
                             t_stop=1. * pq.ms)
        for st in seg.spiketrains:
            self.assertEqual(st.size, 0)

    def test_read_spiketrain_can_return_empty_spiketrain(self):
        """
        Tests if read_spiketrain returns an empty SpikeTrain if no spikes are in
        time range.
        """
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-1256-0.gdf',
                directory=self.local_test_dir, clean=False)
        r = NestIO(filenames=filename)
        st = r.read_spiketrain(gdf_id=0, t_start=400. * pq.ms,
                               t_stop=1. * pq.ms)
        self.assertEqual(st.size, 0)


class TestNestIO_multiple_signal_types(BaseTestIO, unittest.TestCase):
    ioclass = NestIO
    files_to_test = []
    files_to_download = ['0gid-1time-2gex-3Vm-1261-0.dat',
                         '0gid-1time_in_steps-1258-0.gdf']

    def test_read_analogsignal_and_spiketrain(self):
        """
        Test if spiketrains and analogsignals can be read simultaneously
        using read_segment
        """
        files = ['0gid-1time-2gex-3Vm-1261-0.dat',
                 '0gid-1time_in_steps-1258-0.gdf']
        filenames = [get_test_file_full_path(ioclass=NestIO, filename=file,
                                             directory=self.local_test_dir,
                                             clean=False)
                     for file in files]
        r = NestIO(filenames=filenames)
        seg = r.read_segment(gid_list=[], t_start=400 * pq.ms,
                             t_stop=600 * pq.ms,
                             id_column_gdf=0, time_column_gdf=1,
                             id_column_dat=0, time_column_dat=1,
                             value_columns_dat=2)
        self.assertEqual(len(seg.spiketrains), 50)
        self.assertEqual(len(seg.analogsignals), 50)


class TestColumnIO(BaseTestIO, unittest.TestCase):
    ioclass = NestIO
    files_to_download = ['0gid-1time-2Vm-3gex-4gin-1260-0.dat']

    def setUp(self):
        BaseTestIO.setUp(self)
        filename = get_test_file_full_path(
                ioclass=NestIO,
                filename='0gid-1time-2Vm-3gex-4gin-1260-0.dat',
                directory=self.local_test_dir, clean=False)
        self.testIO = ColumnIO(filename=filename)

    def test_no_arguments(self):
        """
        Test if data can be read using the default keyword arguments.
        """
        columns = self.testIO.get_columns()
        expected = self.testIO.data
        np.testing.assert_array_equal(columns, expected)

    def test_single_column_id(self):
        """
        Test if the column_ids keywords works properly.
        """
        column = self.testIO.get_columns(column_ids=1)
        expected = self.testIO.data[:, [1]]
        np.testing.assert_array_equal(column, expected)

    def test_multiple_column_ids(self):
        """
        Test if multiple columns can be read at the same time.
        """
        columns = self.testIO.get_columns(column_ids=range(2))
        expected = self.testIO.data[:, [0, 1]]
        np.testing.assert_array_equal(columns, expected)

    def test_no_condition(self):
        """
        Test if a missing condition function leads to a warning
        """
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            self.testIO.get_columns(condition_column=0)
            # Verify number and content of warning
            assert len(w) == 1
            assert "no condition" in str(w[-1].message)

    def test_no_condition_column(self):
        """
        Test if a missing condition column leads to an error
        """
        with self.assertRaises(ValueError) as context:
            self.testIO.get_columns(condition=lambda x: True)

        self.assertTrue('no condition_column ID provided' in
                        str(context.exception))

    def test_correct_condition_selection(self):
        """
        Test if combination of condition function and condition_column works
        properly.
        """
        condition_column = 0
        condition_function = lambda x: x > 10
        result = self.testIO.get_columns(condition=condition_function,
                                         condition_column=0)
        selected_ids = np.where(condition_function(self.testIO.data[:,
                                                   condition_column]))[0]
        expected = self.testIO.data[selected_ids, :]

        np.testing.assert_array_equal(result, expected)

        assert all(condition_function(result[:, condition_column]))

    def test_sorting(self):
        """
        Test if presorting of columns work properly.
        """
        result = self.testIO.get_columns(sorting_columns=0)

        assert len(result) > 0
        assert all(np.diff(result[:, 0]) >= 0)


if __name__ == "__main__":
    unittest.main()
