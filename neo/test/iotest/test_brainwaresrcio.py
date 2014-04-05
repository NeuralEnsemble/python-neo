# -*- coding: utf-8 -*-
"""
Tests of neo.io.brainwaresrcio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import logging
import os.path
import sys

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

from neo.core import (Block, EventArray, RecordingChannel,
                      RecordingChannelGroup, Segment, SpikeTrain, Unit)
from neo.io import BrainwareSrcIO, brainwaresrcio
from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.tools import (assert_same_sub_schema,
                            assert_neo_object_is_compliant)
from neo.test.iotest.tools import create_generic_reader

PY_VER = sys.version_info[0]

FILES_TO_TEST = ['block_300ms_4rep_1clust_part_ch1.src',
                 'block_500ms_5rep_empty_fullclust_ch1.src',
                 'block_500ms_5rep_empty_partclust_ch1.src',
                 'interleaved_500ms_5rep_ch2.src',
                 'interleaved_500ms_5rep_nospikes_ch1.src',
                 'interleaved_500ms_7rep_noclust_ch1.src',
                 'long_170s_1rep_1clust_ch2.src',
                 'multi_500ms_mulitrep_ch1.src',
                 'random_500ms_12rep_noclust_part_ch2.src',
                 'sequence_500ms_5rep_ch2.src']

FILES_TO_COMPARE = ['block_300ms_4rep_1clust_part_ch1',
                    'block_500ms_5rep_empty_fullclust_ch1',
                    'block_500ms_5rep_empty_partclust_ch1',
                    'interleaved_500ms_5rep_ch2',
                    'interleaved_500ms_5rep_nospikes_ch1',
                    'interleaved_500ms_7rep_noclust_ch1',
                    '',
                    'multi_500ms_mulitrep_ch1',
                    'random_500ms_12rep_noclust_part_ch2',
                    'sequence_500ms_5rep_ch2']


def proc_src(filename):
    '''Load an src file that has already been processed by the official matlab
    file converter.  That matlab data is saved to an m-file, which is then
    converted to a numpy '.npz' file.  This numpy file is the file actually
    loaded.  This function converts it to a neo block and returns the block.
    This block can be compared to the block produced by BrainwareSrcIO to
    make sure BrainwareSrcIO is working properly

    block = proc_src(filename)

    filename: The file name of the numpy file to load.  It should end with
    '*_src_py?.npz'. This will be converted to a neo 'file_origin' property
    with the value '*.src', so the filename to compare should fit that pattern.
    'py?' should be 'py2' for the python 2 version of the numpy file or 'py3'
    for the python 3 version of the numpy file.

    example: filename = 'file1_src_py2.npz'
             src file name = 'file1.src'
    '''
    with np.load(filename) as srcobj:
        srcfile = srcobj.items()[0][1]

    filename = os.path.basename(filename[:-12]+'.src')

    block = Block(file_origin=filename)

    NChannels = srcfile['NChannels'][0, 0][0, 0]
    side = str(srcfile['side'][0, 0][0])
    ADperiod = srcfile['ADperiod'][0, 0][0, 0]

    comm_seg = proc_src_comments(srcfile, filename)
    block.segments.append(comm_seg)

    rcg = proc_src_units(srcfile, filename)
    chan_nums = np.arange(NChannels, dtype='int')
    chan_names = []
    for i in chan_nums:
        name = 'Chan'+str(i)
        chan_names.append(name)
        chan = RecordingChannel(file_origin='filename',
                                name=name,
                                index=int(i))
        rcg.recordingchannels.append(chan)
    rcg.channel_indexes = chan_nums
    rcg.channel_names = np.array(chan_names, dtype='string_')
    block.recordingchannelgroups.append(rcg)

    for rep in srcfile['sets'][0, 0].flatten():
        proc_src_condition(rep, filename, ADperiod, side, block)

    block.create_many_to_one_relationship()

    return block


def proc_src_comments(srcfile, filename):
    '''Get the comments in an src file that has been#!N
    processed by the official
    matlab function.  See proc_src for details'''
    comm_seg = Segment(name='Comments', file_origin=filename)
    commentarray = srcfile['comments'].flatten()[0]
    senders = [res[0] for res in commentarray['sender'].flatten()]
    texts = [res[0] for res in commentarray['text'].flatten()]
    timeStamps = [res[0, 0] for res in commentarray['timeStamp'].flatten()]

    timeStamps = np.array(timeStamps, dtype=np.float32)
    t_start = timeStamps.min()
    timeStamps = pq.Quantity(timeStamps-t_start, units=pq.d).rescale(pq.s)
    texts = np.array(texts, dtype='S')
    senders = np.array(senders, dtype='S')
    t_start = brainwaresrcio.convert_brainwaresrc_timestamp(t_start.tolist())

    comments = EventArray(times=timeStamps,
                          labels=texts,
                          senders=senders)
    comm_seg.eventarrays = [comments]
    comm_seg.rec_datetime = t_start

    return comm_seg


def proc_src_units(srcfile, filename):
    '''Get the units in an src file that has been processed by the official
    matlab function.  See proc_src for details'''
    rcg = RecordingChannelGroup(file_origin=filename)
    un_unit = Unit(name='UnassignedSpikes', file_origin=filename,
                   elliptic=[], boundaries=[], timestamp=[], max_valid=[])

    rcg.units.append(un_unit)

    sortInfo = srcfile['sortInfo'][0, 0]
    timeslice = sortInfo['timeslice'][0, 0]
    maxValid = timeslice['maxValid'][0, 0]
    cluster = timeslice['cluster'][0, 0]
    if len(cluster):
        maxValid = maxValid[0, 0]
        elliptic = [res.flatten() for res in cluster['elliptic'].flatten()]
        boundaries = [res.flatten() for res in cluster['boundaries'].flatten()]
        fullclust = zip(elliptic, boundaries)
        for ielliptic, iboundaries in fullclust:
            unit = Unit(file_origin=filename,
                        boundaries=[iboundaries],
                        elliptic=[ielliptic], timeStamp=[],
                        max_valid=[maxValid])
            rcg.units.append(unit)
    return rcg


def proc_src_condition(rep, filename, ADperiod, side, block):
    '''Get the condition in a src file that has been processed by the official
    matlab function.  See proc_src for details'''

    rcg = block.recordingchannelgroups[0]

    stim = rep['stim'].flatten()
    params = [str(res[0]) for res in stim['paramName'][0].flatten()]
    values = [res for res in stim['paramVal'][0].flatten()]
    stim = dict(zip(params, values))
    sweepLen = rep['sweepLen'][0, 0]

    if not len(rep):
        return

    unassignedSpikes = rep['unassignedSpikes'].flatten()
    if len(unassignedSpikes):
        damaIndexes = [res[0, 0] for res in unassignedSpikes['damaIndex']]
        timeStamps = [res[0, 0] for res in unassignedSpikes['timeStamp']]
        spikeunit = [res.flatten() for res in unassignedSpikes['spikes']]
        respWin = np.array([], dtype=np.int32)
        trains = proc_src_condition_unit(spikeunit, sweepLen, side, ADperiod,
                                         respWin, damaIndexes, timeStamps,
                                         filename)
        rcg.units[0].spiketrains.extend(trains)
        atrains = [trains]
    else:
        damaIndexes = []
        timeStamps = []
        atrains = []

    clusters = rep['clusters'].flatten()
    if len(clusters):
        IdStrings = [res[0] for res in clusters['IdString']]
        sweepLens = [res[0, 0] for res in clusters['sweepLen']]
        respWins = [res.flatten() for res in clusters['respWin']]
        spikeunits = []
        for cluster in clusters['sweeps']:
            if len(cluster):
                spikes = [res.flatten() for res in
                          cluster['spikes'].flatten()]
            else:
                spikes = []
            spikeunits.append(spikes)
    else:
        IdStrings = []
        sweepLens = []
        respWins = []
        spikeunits = []

    for unit, IdString in zip(rcg.units[1:], IdStrings):
        unit.name = str(IdString)

    fullunit = zip(spikeunits, rcg.units[1:], sweepLens, respWins)
    for spikeunit, unit, sweepLen, respWin in fullunit:
        trains = proc_src_condition_unit(spikeunit, sweepLen, side, ADperiod,
                                         respWin, damaIndexes, timeStamps,
                                         filename)
        atrains.append(trains)
        unit.spiketrains.extend(trains)

    atrains = zip(*atrains)
    for trains in atrains:
        segment = Segment(file_origin=filename, feature_type=-1,
                          go_by_closest_unit_center=False,
                          include_unit_bounds=False, **stim)
        block.segments.append(segment)
        segment.spiketrains = trains


def proc_src_condition_unit(spikeunit, sweepLen, side, ADperiod, respWin,
                            damaIndexes, timeStamps, filename):
    '''Get the unit in a condition in a src file that has been processed by
    the official matlab function.  See proc_src for details'''
    if not damaIndexes:
        damaIndexes = [0]*len(spikeunit)
        timeStamps = [0]*len(spikeunit)

    trains = []
    for sweep, damaIndex, timeStamp in zip(spikeunit, damaIndexes,
                                           timeStamps):
        timeStamp = brainwaresrcio.convert_brainwaresrc_timestamp(timeStamp)
        train = proc_src_condition_unit_repetition(sweep, damaIndex,
                                                   timeStamp, sweepLen,
                                                   side, ADperiod, respWin,
                                                   filename)
        trains.append(train)
    return trains


def proc_src_condition_unit_repetition(sweep, damaIndex, timeStamp, sweepLen,
                                       side, ADperiod, respWin, filename):
    '''Get the repetion for a unit in a condition in a src file that has been
    processed by the official matlab function.  See proc_src for details'''
    damaIndex = damaIndex.astype('int32')
    if len(sweep):
        times = np.array([res[0, 0] for res in sweep['time']])
        shapes = np.concatenate([res.flatten()[np.newaxis][np.newaxis] for res
                                 in sweep['shape']], axis=0)
        trig2 = np.array([res[0, 0] for res in sweep['trig2']])
    else:
        times = np.array([])
        shapes = np.array([[[]]])
        trig2 = np.array([])

    times = pq.Quantity(times, units=pq.ms, dtype=np.float32)
    t_start = pq.Quantity(0, units=pq.ms, dtype=np.float32)
    t_stop = pq.Quantity(sweepLen, units=pq.ms, dtype=np.float32)
    trig2 = pq.Quantity(trig2, units=pq.ms, dtype=np.uint8)
    waveforms = pq.Quantity(shapes, dtype=np.int8, units=pq.mV)
    sampling_period = pq.Quantity(ADperiod, units=pq.us)

    train = SpikeTrain(times=times, t_start=t_start, t_stop=t_stop,
                       trig2=trig2, dtype=np.float32, timestamp=timeStamp,
                       dama_index=damaIndex, side=side, copy=True,
                       respwin=respWin, waveforms=waveforms,
                       file_origin=filename)
    train.annotations['side'] = side
    train.sampling_period = sampling_period
    return train


class BrainwareSrcIOTestCase(BaseTestIO, unittest.TestCase):
    '''
    Unit test testcase for neo.io.BrainwareSrcIO
    '''
    ioclass = BrainwareSrcIO
    read_and_write_is_bijective = False

    # These are the files it tries to read and test for compliance
    files_to_test = FILES_TO_TEST

    # these are reference files to compare to
    files_to_compare = FILES_TO_COMPARE

    # add the appropriate suffix depending on the python version
    for i, fname in enumerate(files_to_compare):
        if fname:
            files_to_compare[i] += '_src_py%s.npz' % PY_VER

    # Will fetch from g-node if they don't already exist locally
    # How does it know to do this before any of the other tests?
    files_to_download = files_to_test + files_to_compare

    def setUp(self):
        super(BrainwareSrcIOTestCase, self).setUp()

    def test_reading_same(self):
        for ioobj, path in self.iter_io_objects(return_path=True):
            obj_reader_all = create_generic_reader(ioobj, readall=True)
            obj_reader_base = create_generic_reader(ioobj, target=False)
            obj_reader_next = create_generic_reader(ioobj, target='next_block')
            obj_reader_single = create_generic_reader(ioobj)

            obj_all = obj_reader_all()
            obj_base = obj_reader_base()
            obj_single = obj_reader_single()
            obj_next = [obj_reader_next()]
            while ioobj._isopen:
                obj_next.append(obj_reader_next())

            try:
                assert_same_sub_schema(obj_all[0], obj_base)
                assert_same_sub_schema(obj_all[0], obj_single)
                assert_same_sub_schema(obj_all, obj_next)
            except BaseException as exc:
                exc.args += ('from ' + os.path.basename(path),)
                raise

            self.assertEqual(len(obj_all), len(obj_next))

    def test_against_reference(self):
        for filename, refname in zip(self.files_to_test,
                                     self.files_to_compare):
            if not refname:
                continue
            obj = self.read_file(filename=filename, readall=True)[0]
            refobj = proc_src(self.get_filename_path(refname))
            try:
                assert_neo_object_is_compliant(obj)
                assert_neo_object_is_compliant(refobj)
                assert_same_sub_schema(obj, refobj)
            except BaseException as exc:
                exc.args += ('from ' + filename,)
                raise


if __name__ == '__main__':
    logger = logging.getLogger(BrainwareSrcIO.__module__ +
                               '.' +
                               BrainwareSrcIO.__name__)
    logger.setLevel(100)
    unittest.main()
