# -*- coding: utf-8 -*-
'''
Test to  make sure generated datasets are sane.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from datetime import datetime

import numpy as np
import quantities as pq

from neo.core import (Block, Segment,
                      RecordingChannelGroup, RecordingChannel, Unit,
                      AnalogSignal, AnalogSignalArray,
                      IrregularlySampledSignal, SpikeTrain,
                      Event, Epoch, Spike,
                      EventArray, EpochArray)
from neo.test.generate_datasets import (generate_one_simple_block,
                                               generate_one_simple_segment,
                                               generate_from_supported_objects,
                                               get_fake_value, fake_neo,
                                               TEST_ANNOTATIONS)
from neo.test.tools import assert_arrays_equal, assert_neo_object_is_compliant


class Test__generate_one_simple_segment(unittest.TestCase):
    def test_defaults(self):
        res = generate_one_simple_segment()

        self.assertTrue(isinstance(res, Segment))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.analogsignals), 0)
        self.assertEqual(len(res.analogsignalarrays), 0)
        self.assertEqual(len(res.irregularlysampledsignals), 0)
        self.assertEqual(len(res.spiketrains), 0)
        self.assertEqual(len(res.spikes), 0)
        self.assertEqual(len(res.events), 0)
        self.assertEqual(len(res.epochs), 0)
        self.assertEqual(len(res.eventarrays), 0)
        self.assertEqual(len(res.epocharrays), 0)

    def test_all_supported(self):
        objects = [Block, Segment,
                   RecordingChannelGroup, RecordingChannel, Unit,
                   AnalogSignal, AnalogSignalArray,
                   IrregularlySampledSignal, SpikeTrain,
                   Event, Epoch, Spike,
                   EventArray, EpochArray]

        res = generate_one_simple_segment(supported_objects=objects)

        self.assertTrue(isinstance(res, Segment))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.analogsignals), 4)
        self.assertEqual(len(res.analogsignalarrays), 0)
        self.assertEqual(len(res.irregularlysampledsignals), 0)
        self.assertEqual(len(res.spiketrains), 6)
        self.assertEqual(len(res.spikes), 0)
        self.assertEqual(len(res.events), 0)
        self.assertEqual(len(res.epochs), 0)
        self.assertEqual(len(res.eventarrays), 3)
        self.assertEqual(len(res.epocharrays), 2)

    def test_half_supported(self):
        objects = [Segment,
                   IrregularlySampledSignal, SpikeTrain,
                   Epoch, Spike,
                   EpochArray]

        res = generate_one_simple_segment(supported_objects=objects)

        self.assertTrue(isinstance(res, Segment))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.analogsignals), 0)
        self.assertEqual(len(res.analogsignalarrays), 0)
        self.assertEqual(len(res.irregularlysampledsignals), 0)
        self.assertEqual(len(res.spiketrains), 6)
        self.assertEqual(len(res.spikes), 0)
        self.assertEqual(len(res.events), 0)
        self.assertEqual(len(res.epochs), 0)
        self.assertEqual(len(res.eventarrays), 0)
        self.assertEqual(len(res.epocharrays), 2)

    def test_all_without_block(self):
        objects = [Segment,
                   RecordingChannelGroup, RecordingChannel, Unit,
                   AnalogSignal, AnalogSignalArray,
                   IrregularlySampledSignal, SpikeTrain,
                   Event, Epoch, Spike,
                   EventArray, EpochArray]

        res = generate_one_simple_segment(supported_objects=objects)

        self.assertTrue(isinstance(res, Segment))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.analogsignals), 4)
        self.assertEqual(len(res.analogsignalarrays), 0)
        self.assertEqual(len(res.irregularlysampledsignals), 0)
        self.assertEqual(len(res.spiketrains), 6)
        self.assertEqual(len(res.spikes), 0)
        self.assertEqual(len(res.events), 0)
        self.assertEqual(len(res.epochs), 0)
        self.assertEqual(len(res.eventarrays), 3)
        self.assertEqual(len(res.epocharrays), 2)

    def test_all_without_segment_valueerror(self):
        objects = [Block,
                   RecordingChannelGroup, RecordingChannel, Unit,
                   AnalogSignal, AnalogSignalArray,
                   IrregularlySampledSignal, SpikeTrain,
                   Event, Epoch, Spike,
                   EventArray, EpochArray]

        self.assertRaises(ValueError, generate_one_simple_segment,
                          supported_objects=objects)


class Test__generate_one_simple_block(unittest.TestCase):
    def test_defaults(self):
        res = generate_one_simple_block()

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.segments), 0)

    def test_all_supported(self):
        objects = [Block, Segment,
                   RecordingChannelGroup, RecordingChannel, Unit,
                   AnalogSignal, AnalogSignalArray,
                   IrregularlySampledSignal, SpikeTrain,
                   Event, Epoch, Spike,
                   EventArray, EpochArray]

        res = generate_one_simple_block(supported_objects=objects)

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.segments), 3)
        seg1, seg2, seg3 = res.segments

        self.assertEqual(len(seg1.analogsignals), 4)
        self.assertEqual(len(seg1.analogsignalarrays), 0)
        self.assertEqual(len(seg1.irregularlysampledsignals), 0)
        self.assertEqual(len(seg1.spiketrains), 6)
        self.assertEqual(len(seg1.spikes), 0)
        self.assertEqual(len(seg1.events), 0)
        self.assertEqual(len(seg1.epochs), 0)
        self.assertEqual(len(seg1.eventarrays), 3)
        self.assertEqual(len(seg1.epocharrays), 2)

        self.assertEqual(len(seg2.analogsignals), 4)
        self.assertEqual(len(seg2.analogsignalarrays), 0)
        self.assertEqual(len(seg2.irregularlysampledsignals), 0)
        self.assertEqual(len(seg2.spiketrains), 6)
        self.assertEqual(len(seg2.spikes), 0)
        self.assertEqual(len(seg2.events), 0)
        self.assertEqual(len(seg2.epochs), 0)
        self.assertEqual(len(seg2.eventarrays), 3)
        self.assertEqual(len(seg2.epocharrays), 2)

        self.assertEqual(len(seg3.analogsignals), 4)
        self.assertEqual(len(seg3.analogsignalarrays), 0)
        self.assertEqual(len(seg3.irregularlysampledsignals), 0)
        self.assertEqual(len(seg3.spiketrains), 6)
        self.assertEqual(len(seg3.spikes), 0)
        self.assertEqual(len(seg3.events), 0)
        self.assertEqual(len(seg3.epochs), 0)
        self.assertEqual(len(seg3.eventarrays), 3)
        self.assertEqual(len(seg3.epocharrays), 2)

    def test_half_supported(self):
        objects = [Block, Segment,
                   IrregularlySampledSignal, SpikeTrain,
                   Epoch, Spike,
                   EpochArray]

        res = generate_one_simple_block(supported_objects=objects)

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.segments), 3)
        seg1, seg2, seg3 = res.segments

        self.assertEqual(len(seg1.analogsignals), 0)
        self.assertEqual(len(seg1.analogsignalarrays), 0)
        self.assertEqual(len(seg1.irregularlysampledsignals), 0)
        self.assertEqual(len(seg1.spiketrains), 6)
        self.assertEqual(len(seg1.spikes), 0)
        self.assertEqual(len(seg1.events), 0)
        self.assertEqual(len(seg1.epochs), 0)
        self.assertEqual(len(seg1.eventarrays), 0)
        self.assertEqual(len(seg1.epocharrays), 2)

        self.assertEqual(len(seg2.analogsignals), 0)
        self.assertEqual(len(seg2.analogsignalarrays), 0)
        self.assertEqual(len(seg2.irregularlysampledsignals), 0)
        self.assertEqual(len(seg2.spiketrains), 6)
        self.assertEqual(len(seg2.spikes), 0)
        self.assertEqual(len(seg2.events), 0)
        self.assertEqual(len(seg2.epochs), 0)
        self.assertEqual(len(seg2.eventarrays), 0)
        self.assertEqual(len(seg2.epocharrays), 2)

        self.assertEqual(len(seg3.analogsignals), 0)
        self.assertEqual(len(seg3.analogsignalarrays), 0)
        self.assertEqual(len(seg3.irregularlysampledsignals), 0)
        self.assertEqual(len(seg3.spiketrains), 6)
        self.assertEqual(len(seg3.spikes), 0)
        self.assertEqual(len(seg3.events), 0)
        self.assertEqual(len(seg3.epochs), 0)
        self.assertEqual(len(seg3.eventarrays), 0)
        self.assertEqual(len(seg3.epocharrays), 2)

    def test_all_without_segment(self):
        objects = [Block,
                   RecordingChannelGroup, RecordingChannel, Unit,
                   AnalogSignal, AnalogSignalArray,
                   IrregularlySampledSignal, SpikeTrain,
                   Event, Epoch, Spike,
                   EventArray, EpochArray]

        res = generate_one_simple_block(supported_objects=objects)

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.segments), 0)

    def test_all_without_block_valueerror(self):
        objects = [Segment,
                   RecordingChannelGroup, RecordingChannel, Unit,
                   AnalogSignal, AnalogSignalArray,
                   IrregularlySampledSignal, SpikeTrain,
                   Event, Epoch, Spike,
                   EventArray, EpochArray]

        self.assertRaises(ValueError, generate_one_simple_block,
                          supported_objects=objects)


class Test__generate_from_supported_objects(unittest.TestCase):
    def test_no_object_valueerror(self):
        objects = []

        self.assertRaises(ValueError, generate_from_supported_objects, objects)

    def test_all(self):
        objects = [Block, Segment,
                   RecordingChannelGroup, RecordingChannel, Unit,
                   AnalogSignal, AnalogSignalArray,
                   IrregularlySampledSignal, SpikeTrain,
                   Event, Epoch, Spike,
                   EventArray, EpochArray]

        res = generate_from_supported_objects(objects)

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.segments), 3)
        seg1, seg2, seg3 = res.segments

        self.assertEqual(len(seg1.analogsignals), 4)
        self.assertEqual(len(seg1.analogsignalarrays), 0)
        self.assertEqual(len(seg1.irregularlysampledsignals), 0)
        self.assertEqual(len(seg1.spiketrains), 6)
        self.assertEqual(len(seg1.spikes), 0)
        self.assertEqual(len(seg1.events), 0)
        self.assertEqual(len(seg1.epochs), 0)
        self.assertEqual(len(seg1.eventarrays), 3)
        self.assertEqual(len(seg1.epocharrays), 2)

        self.assertEqual(len(seg2.analogsignals), 4)
        self.assertEqual(len(seg2.analogsignalarrays), 0)
        self.assertEqual(len(seg2.irregularlysampledsignals), 0)
        self.assertEqual(len(seg2.spiketrains), 6)
        self.assertEqual(len(seg2.spikes), 0)
        self.assertEqual(len(seg2.events), 0)
        self.assertEqual(len(seg2.epochs), 0)
        self.assertEqual(len(seg2.eventarrays), 3)
        self.assertEqual(len(seg2.epocharrays), 2)

        self.assertEqual(len(seg3.analogsignals), 4)
        self.assertEqual(len(seg3.analogsignalarrays), 0)
        self.assertEqual(len(seg3.irregularlysampledsignals), 0)
        self.assertEqual(len(seg3.spiketrains), 6)
        self.assertEqual(len(seg3.spikes), 0)
        self.assertEqual(len(seg3.events), 0)
        self.assertEqual(len(seg3.epochs), 0)
        self.assertEqual(len(seg3.eventarrays), 3)
        self.assertEqual(len(seg3.epocharrays), 2)

    def test_block(self):
        objects = [Block]

        res = generate_from_supported_objects(objects)

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.segments), 0)

    def test_block_segment(self):
        objects = [Segment, Block]

        res = generate_from_supported_objects(objects)

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.segments), 3)
        seg1, seg2, seg3 = res.segments

        self.assertEqual(len(seg1.analogsignals), 0)
        self.assertEqual(len(seg1.analogsignalarrays), 0)
        self.assertEqual(len(seg1.irregularlysampledsignals), 0)
        self.assertEqual(len(seg1.spiketrains), 0)
        self.assertEqual(len(seg1.spikes), 0)
        self.assertEqual(len(seg1.events), 0)
        self.assertEqual(len(seg1.epochs), 0)
        self.assertEqual(len(seg1.eventarrays), 0)
        self.assertEqual(len(seg1.epocharrays), 0)

        self.assertEqual(len(seg2.analogsignals), 0)
        self.assertEqual(len(seg2.analogsignalarrays), 0)
        self.assertEqual(len(seg2.irregularlysampledsignals), 0)
        self.assertEqual(len(seg2.spiketrains), 0)
        self.assertEqual(len(seg2.spikes), 0)
        self.assertEqual(len(seg2.events), 0)
        self.assertEqual(len(seg2.epochs), 0)
        self.assertEqual(len(seg2.eventarrays), 0)
        self.assertEqual(len(seg2.epocharrays), 0)

        self.assertEqual(len(seg3.analogsignals), 0)
        self.assertEqual(len(seg3.analogsignalarrays), 0)
        self.assertEqual(len(seg3.irregularlysampledsignals), 0)
        self.assertEqual(len(seg3.spiketrains), 0)
        self.assertEqual(len(seg3.spikes), 0)
        self.assertEqual(len(seg3.events), 0)
        self.assertEqual(len(seg3.epochs), 0)
        self.assertEqual(len(seg3.eventarrays), 0)
        self.assertEqual(len(seg3.epocharrays), 0)

    def test_segment(self):
        objects = [Segment]

        res = generate_from_supported_objects(objects)

        self.assertTrue(isinstance(res, Segment))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.analogsignals), 0)
        self.assertEqual(len(res.analogsignalarrays), 0)
        self.assertEqual(len(res.irregularlysampledsignals), 0)
        self.assertEqual(len(res.spiketrains), 0)
        self.assertEqual(len(res.spikes), 0)
        self.assertEqual(len(res.events), 0)
        self.assertEqual(len(res.epochs), 0)
        self.assertEqual(len(res.eventarrays), 0)
        self.assertEqual(len(res.epocharrays), 0)

    def test_all_without_block(self):
        objects = [Segment,
                   RecordingChannelGroup, RecordingChannel, Unit,
                   AnalogSignal, AnalogSignalArray,
                   IrregularlySampledSignal, SpikeTrain,
                   Event, Epoch, Spike,
                   EventArray, EpochArray]

        res = generate_from_supported_objects(objects)

        self.assertTrue(isinstance(res, Segment))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.analogsignals), 4)
        self.assertEqual(len(res.analogsignalarrays), 0)
        self.assertEqual(len(res.irregularlysampledsignals), 0)
        self.assertEqual(len(res.spiketrains), 6)
        self.assertEqual(len(res.spikes), 0)
        self.assertEqual(len(res.events), 0)
        self.assertEqual(len(res.epochs), 0)
        self.assertEqual(len(res.eventarrays), 3)
        self.assertEqual(len(res.epocharrays), 2)

    def test_all_without_segment(self):
        objects = [Block,
                   RecordingChannelGroup, RecordingChannel, Unit,
                   AnalogSignal, AnalogSignalArray,
                   IrregularlySampledSignal, SpikeTrain,
                   Event, Epoch, Spike,
                   EventArray, EpochArray]

        res = generate_from_supported_objects(supported_objects=objects)

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.segments), 0)


class Test__get_fake_value(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test__t_start(self):
        name = 't_start'
        datatype = pq.Quantity
        targ = 0.0 * pq.millisecond

        res = get_fake_value(name, datatype)
        self.assertTrue(isinstance(res, pq.Quantity))
        self.assertEqual(res.units, pq.millisecond)
        assert_arrays_equal(targ, res)

        self.assertRaises(ValueError, get_fake_value, name, datatype, dim=1)
        self.assertRaises(ValueError, get_fake_value, name, np.ndarray)

    def test__t_stop(self):
        name = 't_stop'
        datatype = pq.Quantity
        targ = 1.0 * pq.millisecond

        res = get_fake_value(name, datatype)
        self.assertTrue(isinstance(res, pq.Quantity))
        self.assertEqual(res.units, pq.millisecond)
        assert_arrays_equal(targ, res)

        self.assertRaises(ValueError, get_fake_value, name, datatype, dim=1)
        self.assertRaises(ValueError, get_fake_value, name, np.ndarray)

    def test__sampling_rate(self):
        name = 'sampling_rate'
        datatype = pq.Quantity
        targ = 10000.0 * pq.Hz

        res = get_fake_value(name, datatype)
        self.assertTrue(isinstance(res, pq.Quantity))
        self.assertEqual(res.units, pq.Hz)
        assert_arrays_equal(targ, res)

        self.assertRaises(ValueError, get_fake_value, name, datatype, dim=1)
        self.assertRaises(ValueError, get_fake_value, name, np.ndarray)

    def test__str(self):
        name = 'test__str'
        datatype = str
        targ = str(np.random.randint(100000))

        res = get_fake_value(name, datatype, seed=0)
        self.assertTrue(isinstance(res, str))
        self.assertEqual(targ, res)

    def test__int(self):
        name = 'test__int'
        datatype = int
        targ = np.random.randint(100)

        res = get_fake_value(name, datatype, seed=0)
        self.assertTrue(isinstance(res, int))
        self.assertEqual(targ, res)

    def test__float(self):
        name = 'test__float'
        datatype = float
        targ = 1000. * np.random.random()

        res = get_fake_value(name, datatype, seed=0)
        self.assertTrue(isinstance(res, float))
        self.assertEqual(targ, res)

    def test__datetime(self):
        name = 'test__datetime'
        datatype = datetime

        res = get_fake_value(name, datatype)
        self.assertTrue(isinstance(res, datetime))

    def test__quantity(self):
        name = 'test__quantity'
        datatype = pq.Quantity
        dim = 2

        size = []
        for i in range(int(dim)):
            size.append(np.random.randint(100) + 1)
        targ = np.random.random(size) * pq.millisecond

        res = get_fake_value(name, datatype, dim=dim, seed=0)
        self.assertTrue(isinstance(res, pq.Quantity))
        self.assertEqual(res.units, pq.millisecond)
        assert_arrays_equal(targ, res)

    def test__ndarray(self):
        name = 'test__quantity'
        datatype = np.ndarray
        dim = 2

        size = []
        for i in range(int(dim)):
            size.append(np.random.randint(100) + 1)
        targ = np.random.random(size) * pq.millisecond

        res = get_fake_value(name, datatype, dim=dim, seed=0)
        self.assertTrue(isinstance(res, np.ndarray))
        assert_arrays_equal(targ, res)

    def test__other_valueerror(self):
        name = 'test__other_fail'
        datatype = set([1, 2, 3, 4])

        self.assertRaises(ValueError, get_fake_value, name, datatype)


class Test__generate_datasets(unittest.TestCase):
    def setUp(self):
        self.annotations = dict([(str(x), TEST_ANNOTATIONS[x]) for x in
                                 range(len(TEST_ANNOTATIONS))])

    def test__block__cascade(self):
        obj_type = 'Block'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.segments), 1)
        seg = res.segments[0]
        self.assertEqual(seg.annotations, self.annotations)

        self.assertEqual(len(res.recordingchannelgroups), 1)
        rcg = res.recordingchannelgroups[0]
        self.assertEqual(rcg.annotations, self.annotations)

        self.assertEqual(len(seg.analogsignalarrays), 1)
        self.assertEqual(len(seg.analogsignals), 1)
        self.assertEqual(len(seg.irregularlysampledsignals), 1)
        self.assertEqual(len(seg.spiketrains), 1)
        self.assertEqual(len(seg.spikes), 1)
        self.assertEqual(len(seg.events), 1)
        self.assertEqual(len(seg.epochs), 1)
        self.assertEqual(len(seg.eventarrays), 1)
        self.assertEqual(len(seg.epocharrays), 1)
        self.assertEqual(seg.analogsignalarrays[0].annotations,
                         self.annotations)
        self.assertEqual(seg.analogsignals[0].annotations,
                         self.annotations)
        self.assertEqual(seg.irregularlysampledsignals[0].annotations,
                         self.annotations)
        self.assertEqual(seg.spiketrains[0].annotations,
                         self.annotations)
        self.assertEqual(seg.spikes[0].annotations,
                         self.annotations)
        self.assertEqual(seg.events[0].annotations,
                         self.annotations)
        self.assertEqual(seg.epochs[0].annotations,
                         self.annotations)
        self.assertEqual(seg.eventarrays[0].annotations,
                         self.annotations)
        self.assertEqual(seg.epocharrays[0].annotations,
                         self.annotations)

        self.assertEqual(len(rcg.recordingchannels), 1)
        rchan = rcg.recordingchannels[0]
        self.assertEqual(rchan.annotations, self.annotations)

        self.assertEqual(len(rcg.units), 1)
        unit = rcg.units[0]
        self.assertEqual(unit.annotations, self.annotations)

        self.assertEqual(len(rcg.analogsignalarrays), 1)
        self.assertEqual(rcg.analogsignalarrays[0].annotations,
                         self.annotations)

        self.assertEqual(len(rchan.analogsignals), 1)
        self.assertEqual(len(rchan.irregularlysampledsignals), 1)
        self.assertEqual(rchan.analogsignals[0].annotations,
                         self.annotations)
        self.assertEqual(rchan.irregularlysampledsignals[0].annotations,
                         self.annotations)

        self.assertEqual(len(unit.spiketrains), 1)
        self.assertEqual(len(unit.spikes), 1)
        self.assertEqual(unit.spiketrains[0].annotations,
                         self.annotations)
        self.assertEqual(unit.spikes[0].annotations,
                         self.annotations)

    def test__block__nocascade(self):
        obj_type = 'Block'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.segments), 0)
        self.assertEqual(len(res.recordingchannelgroups), 0)

    def test__segment__cascade(self):
        obj_type = 'Segment'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Segment))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.analogsignalarrays), 1)
        self.assertEqual(len(res.analogsignals), 1)
        self.assertEqual(len(res.irregularlysampledsignals), 1)
        self.assertEqual(len(res.spiketrains), 1)
        self.assertEqual(len(res.spikes), 1)
        self.assertEqual(len(res.events), 1)
        self.assertEqual(len(res.epochs), 1)
        self.assertEqual(len(res.eventarrays), 1)
        self.assertEqual(len(res.epocharrays), 1)

        self.assertEqual(res.analogsignalarrays[0].annotations,
                         self.annotations)
        self.assertEqual(res.analogsignals[0].annotations,
                         self.annotations)
        self.assertEqual(res.irregularlysampledsignals[0].annotations,
                         self.annotations)
        self.assertEqual(res.spiketrains[0].annotations,
                         self.annotations)
        self.assertEqual(res.spikes[0].annotations,
                         self.annotations)
        self.assertEqual(res.events[0].annotations,
                         self.annotations)
        self.assertEqual(res.epochs[0].annotations,
                         self.annotations)
        self.assertEqual(res.eventarrays[0].annotations,
                         self.annotations)
        self.assertEqual(res.epocharrays[0].annotations,
                         self.annotations)

    def test__segment__nocascade(self):
        obj_type = 'Segment'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Segment))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.analogsignalarrays), 0)
        self.assertEqual(len(res.analogsignals), 0)
        self.assertEqual(len(res.irregularlysampledsignals), 0)
        self.assertEqual(len(res.spiketrains), 0)
        self.assertEqual(len(res.spikes), 0)
        self.assertEqual(len(res.events), 0)
        self.assertEqual(len(res.epochs), 0)
        self.assertEqual(len(res.eventarrays), 0)
        self.assertEqual(len(res.epocharrays), 0)

    def test__recordingchannelgroup__cascade(self):
        obj_type = 'RecordingChannelGroup'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, RecordingChannelGroup))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.recordingchannels), 1)
        rchan = res.recordingchannels[0]
        self.assertEqual(rchan.annotations, self.annotations)

        self.assertEqual(len(res.units), 1)
        unit = res.units[0]
        self.assertEqual(unit.annotations, self.annotations)

        self.assertEqual(len(res.analogsignalarrays), 1)
        self.assertEqual(res.analogsignalarrays[0].annotations,
                         self.annotations)

        self.assertEqual(len(rchan.analogsignals), 1)
        self.assertEqual(len(rchan.irregularlysampledsignals), 1)
        self.assertEqual(rchan.analogsignals[0].annotations,
                         self.annotations)
        self.assertEqual(rchan.irregularlysampledsignals[0].annotations,
                         self.annotations)

        self.assertEqual(len(unit.spiketrains), 1)
        self.assertEqual(len(unit.spikes), 1)
        self.assertEqual(unit.spiketrains[0].annotations,
                         self.annotations)
        self.assertEqual(unit.spikes[0].annotations,
                         self.annotations)

    def test__recordingchannelgroup__nocascade(self):
        obj_type = 'RecordingChannelGroup'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, RecordingChannelGroup))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.recordingchannels), 0)
        self.assertEqual(len(res.units), 0)
        self.assertEqual(len(res.analogsignalarrays), 0)

    def test__recordingchannel__cascade(self):
        obj_type = 'RecordingChannel'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, RecordingChannel))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.analogsignals), 1)
        self.assertEqual(len(res.irregularlysampledsignals), 1)

        self.assertEqual(res.analogsignals[0].annotations,
                         self.annotations)
        self.assertEqual(res.irregularlysampledsignals[0].annotations,
                         self.annotations)

    def test__recordingchannel__nocascade(self):
        obj_type = 'RecordingChannel'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, RecordingChannel))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.analogsignals), 0)
        self.assertEqual(len(res.irregularlysampledsignals), 0)

    def test__unit__cascade(self):
        obj_type = 'Unit'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Unit))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.spiketrains), 1)
        self.assertEqual(len(res.spikes), 1)

        self.assertEqual(res.spiketrains[0].annotations,
                         self.annotations)
        self.assertEqual(res.spikes[0].annotations,
                         self.annotations)

    def test__unit__nocascade(self):
        obj_type = 'Unit'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Unit))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        self.assertEqual(len(res.spiketrains), 0)
        self.assertEqual(len(res.spikes), 0)

    def test__analogsignal__cascade(self):
        obj_type = 'AnalogSignal'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, AnalogSignal))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__analogsignal__nocascade(self):
        obj_type = 'AnalogSignal'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, AnalogSignal))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__analogsignalarray__cascade(self):
        obj_type = 'AnalogSignalArray'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, AnalogSignalArray))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__analogsignalarray__nocascade(self):
        obj_type = 'AnalogSignalArray'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, AnalogSignalArray))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__irregularlysampledsignal__cascade(self):
        obj_type = 'IrregularlySampledSignal'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, IrregularlySampledSignal))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__irregularlysampledsignal__nocascade(self):
        obj_type = 'IrregularlySampledSignal'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, IrregularlySampledSignal))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__spiketrain__cascade(self):
        obj_type = 'SpikeTrain'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, SpikeTrain))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__spiketrain__nocascade(self):
        obj_type = 'SpikeTrain'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, SpikeTrain))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__event__cascade(self):
        obj_type = 'Event'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Event))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__event__nocascade(self):
        obj_type = 'Event'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Event))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__epoch__cascade(self):
        obj_type = 'Epoch'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Epoch))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__epoch__nocascade(self):
        obj_type = 'Epoch'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Epoch))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__spike__cascade(self):
        obj_type = 'Spike'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Spike))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__spike__nocascade(self):
        obj_type = 'Spike'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, Spike))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__eventarray__cascade(self):
        obj_type = 'EventArray'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, EventArray))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__eventarray__nocascade(self):
        obj_type = 'EventArray'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, EventArray))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__epocharray__cascade(self):
        obj_type = 'EpochArray'
        cascade = True
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, EpochArray))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

    def test__epocharray__nocascade(self):
        obj_type = 'EpochArray'
        cascade = False
        res = fake_neo(obj_type=obj_type, cascade=cascade)

        self.assertTrue(isinstance(res, EpochArray))
        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)
