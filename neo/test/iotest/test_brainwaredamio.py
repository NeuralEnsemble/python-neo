# -*- coding: utf-8 -*-
"""
Tests of neo.io.brainwaredamio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import os.path
import sys

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

from neo.core import (AnalogSignal, Block, RecordingChannel,
                      RecordingChannelGroup, Segment)
from neo.io import BrainwareDamIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.tools import (assert_same_sub_schema,
                            assert_neo_object_is_compliant)
from neo.test.iotest.tools import create_generic_reader

PY_VER = sys.version_info[0]


def proc_dam(filename):
    '''Load an dam file that has already been processed by the official matlab
    file converter.  That matlab data is saved to an m-file, which is then
    converted to a numpy '.npz' file.  This numpy file is the file actually
    loaded.  This function converts it to a neo block and returns the block.
    This block can be compared to the block produced by BrainwareDamIO to
    make sure BrainwareDamIO is working properly

    block = proc_dam(filename)

    filename: The file name of the numpy file to load.  It should end with
    '*_dam_py?.npz'. This will be converted to a neo 'file_origin' property
    with the value '*.dam', so the filename to compare should fit that pattern.
    'py?' should be 'py2' for the python 2 version of the numpy file or 'py3'
    for the python 3 version of the numpy file.

    example: filename = 'file1_dam_py2.npz'
             dam file name = 'file1.dam'
    '''
    with np.load(filename) as damobj:
        damfile = damobj.items()[0][1].flatten()

    filename = os.path.basename(filename[:-12]+'.dam')

    signals = [res.flatten() for res in damfile['signal']]
    stimIndexes = [int(res[0, 0].tolist()) for res in damfile['stimIndex']]
    timestamps = [res[0, 0] for res in damfile['timestamp']]

    block = Block(file_origin=filename)

    rcg = RecordingChannelGroup(file_origin=filename)
    chan = RecordingChannel(file_origin=filename, index=0, name='Chan1')
    rcg.channel_indexes = np.array([1])
    rcg.channel_names = np.array(['Chan1'], dtype='S')

    block.recordingchannelgroups.append(rcg)
    rcg.recordingchannels.append(chan)

    params = [res['params'][0, 0].flatten() for res in damfile['stim']]
    values = [res['values'][0, 0].flatten() for res in damfile['stim']]
    params = [[res1[0] for res1 in res] for res in params]
    values = [[res1 for res1 in res] for res in values]
    stims = [dict(zip(param, value)) for param, value in zip(params, values)]

    fulldam = zip(stimIndexes, timestamps, signals, stims)
    for stimIndex, timestamp, signal, stim in fulldam:
        sig = AnalogSignal(signal=signal*pq.mV,
                           t_start=timestamp*pq.d,
                           file_origin=filename,
                           sampling_period=1.*pq.s)
        segment = Segment(file_origin=filename,
                          index=stimIndex,
                          **stim)
        segment.analogsignals = [sig]
        block.segments.append(segment)

    block.create_many_to_one_relationship()

    return block


class BrainwareDamIOTestCase(BaseTestIO, unittest.TestCase):
    '''
    Unit test testcase for neo.io.BrainwareDamIO
    '''
    ioclass = BrainwareDamIO
    read_and_write_is_bijective = False

    # These are the files it tries to read and test for compliance
    files_to_test = ['block_300ms_4rep_1clust_part_ch1.dam',
                     'interleaved_500ms_5rep_ch2.dam',
                     'long_170s_1rep_1clust_ch2.dam',
                     'multi_500ms_mulitrep_ch1.dam',
                     'random_500ms_12rep_noclust_part_ch2.dam',
                     'sequence_500ms_5rep_ch2.dam']

    # these are reference files to compare to
    files_to_compare = ['block_300ms_4rep_1clust_part_ch1',
                        'interleaved_500ms_5rep_ch2',
                        '',
                        'multi_500ms_mulitrep_ch1',
                        'random_500ms_12rep_noclust_part_ch2',
                        'sequence_500ms_5rep_ch2']

    # add the appropriate suffix depending on the python version
    for i, fname in enumerate(files_to_compare):
        if fname:
            files_to_compare[i] += '_dam_py%s.npz' % PY_VER

    # Will fetch from g-node if they don't already exist locally
    # How does it know to do this before any of the other tests?
    files_to_download = files_to_test + files_to_compare

    def test_reading_same(self):
        for ioobj, path in self.iter_io_objects(return_path=True):
            obj_reader_base = create_generic_reader(ioobj, target=False)
            obj_reader_single = create_generic_reader(ioobj)

            obj_base = obj_reader_base()
            obj_single = obj_reader_single()

            try:
                assert_same_sub_schema(obj_base, obj_single)
            except BaseException as exc:
                exc.args += ('from ' + os.path.basename(path),)
                raise

    def test_against_reference(self):
        for filename, refname in zip(self.files_to_test,
                                     self.files_to_compare):
            if not refname:
                continue
            obj = self.read_file(filename=filename)
            refobj = proc_dam(self.get_filename_path(refname))

            try:
                assert_neo_object_is_compliant(obj)
                assert_neo_object_is_compliant(refobj)
                assert_same_sub_schema(obj, refobj)
            except BaseException as exc:
                exc.args += ('from ' + filename,)
                raise


if __name__ == '__main__':
    unittest.main()
