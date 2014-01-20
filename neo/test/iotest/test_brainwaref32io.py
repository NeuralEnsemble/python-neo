# -*- coding: utf-8 -*-
"""
Tests of neo.io.brainwaref32io
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

from neo.core import Block, RecordingChannelGroup, Segment, SpikeTrain, Unit
from neo.io import BrainwareF32IO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.tools import (assert_same_sub_schema,
                            assert_neo_object_is_compliant)
from neo.test.iotest.tools import create_generic_reader

PY_VER = sys.version_info[0]


def proc_f32(filename):
    '''Load an f32 file that has already been processed by the official matlab
    file converter.  That matlab data is saved to an m-file, which is then
    converted to a numpy '.npz' file.  This numpy file is the file actually
    loaded.  This function converts it to a neo block and returns the block.
    This block can be compared to the block produced by BrainwareF32IO to
    make sure BrainwareF32IO is working properly

    block = proc_f32(filename)

    filename: The file name of the numpy file to load.  It should end with
    '*_f32_py?.npz'. This will be converted to a neo 'file_origin' property
    with the value '*.f32', so the filename to compare should fit that pattern.
    'py?' should be 'py2' for the python 2 version of the numpy file or 'py3'
    for the python 3 version of the numpy file.

    example: filename = 'file1_f32_py2.npz'
             f32 file name = 'file1.f32'
    '''

    filenameorig = os.path.basename(filename[:-12]+'.f32')

    # create the objects to store other objects
    block = Block(file_origin=filenameorig)
    rcg = RecordingChannelGroup(file_origin=filenameorig)
    rcg.channel_indexes = np.array([], dtype=np.int)
    rcg.channel_names = np.array([], dtype='S')
    unit = Unit(file_origin=filenameorig)

    # load objects into their containers
    block.recordingchannelgroups.append(rcg)
    rcg.units.append(unit)

    try:
        with np.load(filename) as f32obj:
            f32file = f32obj.items()[0][1].flatten()
    except IOError as exc:
        if 'as a pickle' in exc.message:
            block.create_many_to_one_relationship()
            return block
        else:
            raise

    sweeplengths = [res[0, 0].tolist() for res in f32file['sweeplength']]
    stims = [res.flatten().tolist() for res in f32file['stim']]

    sweeps = [res['spikes'].flatten() for res in f32file['sweep'] if res.size]

    fullf32 = zip(sweeplengths, stims, sweeps)
    for sweeplength, stim, sweep in fullf32:
        for trainpts in sweep:
            if trainpts.size:
                trainpts = trainpts.flatten().astype('float32')
            else:
                trainpts = []

            paramnames = ['Param%s' % i for i in range(len(stim))]
            params = dict(zip(paramnames, stim))
            train = SpikeTrain(trainpts, units=pq.ms,
                               t_start=0, t_stop=sweeplength,
                               file_origin=filenameorig)

            segment = Segment(file_origin=filenameorig, **params)
            segment.spiketrains = [train]
            unit.spiketrains.append(train)
            block.segments.append(segment)

    block.create_many_to_one_relationship()

    return block


class BrainwareF32IOTestCase(BaseTestIO, unittest.TestCase):
    '''
    Unit test testcase for neo.io.BrainwareF32IO
    '''
    ioclass = BrainwareF32IO
    read_and_write_is_bijective = False

    # These are the files it tries to read and test for compliance
    files_to_test = ['block_300ms_4rep_1clust_part_ch1.f32',
                     'block_500ms_5rep_empty_fullclust_ch1.f32',
                     'block_500ms_5rep_empty_partclust_ch1.f32',
                     'interleaved_500ms_5rep_ch2.f32',
                     'interleaved_500ms_5rep_nospikes_ch1.f32',
                     'multi_500ms_mulitrep_ch1.f32',
                     'random_500ms_12rep_noclust_part_ch2.f32',
                     'sequence_500ms_5rep_ch2.f32']

    # add the appropriate suffix depending on the python version
    suffix = '_f32_py%s.npz' % PY_VER
    files_to_download = files_to_test[:]

    # add the reference files to the list of files to download
    files_to_compare = []
    for fname in files_to_test:
        if fname:
            files_to_compare.append(os.path.splitext(fname)[0] + suffix)

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
        for obj, path in self.iter_objects(return_path=True):
            filename = os.path.basename(path)

            refpath = os.path.splitext(path)[0] + self.suffix
            refobj = proc_f32(refpath)

            try:
                assert_neo_object_is_compliant(obj)
                assert_neo_object_is_compliant(refobj)
                assert_same_sub_schema(obj, refobj)
            except BaseException as exc:
                exc.args += ('from ' + filename,)
                raise


if __name__ == '__main__':
    unittest.main()
