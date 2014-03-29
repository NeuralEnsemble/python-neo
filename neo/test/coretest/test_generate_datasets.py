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

from neo.core import (class_by_name, Block, Segment,
                      RecordingChannelGroup, RecordingChannel, Unit,
                      AnalogSignal, AnalogSignalArray,
                      IrregularlySampledSignal, SpikeTrain,
                      Event, Epoch, Spike,
                      EventArray, EpochArray)
from neo.test.generate_datasets import (generate_one_simple_block,
                                        generate_one_simple_segment,
                                        generate_from_supported_objects,
                                        get_fake_value, get_fake_values,
                                        fake_neo, TEST_ANNOTATIONS)
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

    def test__name(self):
        name = 'name'
        datatype = str
        obj = 'Block'
        targ = 'Block'+str(np.random.randint(100000))

        res = get_fake_value(name, datatype, seed=0, obj=obj)
        self.assertTrue(isinstance(res, str))
        self.assertEqual(targ, res)

        self.assertRaises(ValueError, get_fake_value, name, datatype, dim=1)
        self.assertRaises(ValueError, get_fake_value, name, np.ndarray)

    def test__description(self):
        name = 'description'
        datatype = str
        obj = Segment
        targ = 'test Segment '+str(np.random.randint(100000))

        res = get_fake_value(name, datatype, seed=0, obj=obj)
        self.assertTrue(isinstance(res, str))
        self.assertEqual(targ, res)

        self.assertRaises(ValueError, get_fake_value, name, datatype, dim=1)
        self.assertRaises(ValueError, get_fake_value, name, np.ndarray)

    def test__file_origin(self):
        name = 'file_origin'
        datatype = str
        targ = 'test_file.txt'

        res = get_fake_value(name, datatype, seed=0)
        self.assertTrue(isinstance(res, str))
        self.assertEqual(targ, res)

        self.assertRaises(ValueError, get_fake_value, name, datatype, dim=1)
        self.assertRaises(ValueError, get_fake_value, name, np.ndarray)

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
        targ = datetime.fromtimestamp(1000000000*np.random.random())

        res = get_fake_value(name, datatype, seed=0)
        self.assertTrue(isinstance(res, datetime))
        self.assertEqual(res, targ)

    def test__quantity(self):
        name = 'test__quantity'
        datatype = pq.Quantity
        dim = 2

        size = []
        units = np.random.choice(['nA', 'mA', 'A', 'mV', 'V'])
        for i in range(int(dim)):
            size.append(np.random.randint(5) + 1)
        targ = pq.Quantity(np.random.random(size)*1000, units=units)

        res = get_fake_value(name, datatype, dim=dim, seed=0)
        self.assertTrue(isinstance(res, pq.Quantity))
        self.assertEqual(res.units, getattr(pq, units))
        assert_arrays_equal(targ, res)

    def test__quantity_force_units(self):
        name = 'test__quantity'
        datatype = np.ndarray
        dim = 2
        units = pq.ohm

        size = []
        for i in range(int(dim)):
            size.append(np.random.randint(5) + 1)
        targ = pq.Quantity(np.random.random(size)*1000, units=units)

        res = get_fake_value(name, datatype, dim=dim, seed=0, units=units)
        self.assertTrue(isinstance(res, np.ndarray))
        assert_arrays_equal(targ, res)

    def test__ndarray(self):
        name = 'test__ndarray'
        datatype = np.ndarray
        dim = 2

        size = []
        for i in range(int(dim)):
            size.append(np.random.randint(5) + 1)
        targ = np.random.random(size)*1000

        res = get_fake_value(name, datatype, dim=dim, seed=0)
        self.assertTrue(isinstance(res, np.ndarray))
        assert_arrays_equal(targ, res)

    def test__list(self):
        name = 'test__list'
        datatype = list
        dim = 2

        size = []
        for i in range(int(dim)):
            size.append(np.random.randint(5) + 1)
        targ = (np.random.random(size)*1000).tolist()

        res = get_fake_value(name, datatype, dim=dim, seed=0)
        self.assertTrue(isinstance(res, list))
        self.assertEqual(targ, res)

    def test__other_valueerror(self):
        name = 'test__other_fail'
        datatype = set([1, 2, 3, 4])

        self.assertRaises(ValueError, get_fake_value, name, datatype)


class Test__get_fake_values(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.annotations = dict([(str(x), TEST_ANNOTATIONS[x]) for x in
                                 range(len(TEST_ANNOTATIONS))])
        self.annotations['seed'] = 0

    def subcheck__get_fake_values(self, cls):
        res1 = get_fake_values(cls, annotate=False, seed=0)
        res2 = get_fake_values(cls, annotate=True, seed=0)

        if hasattr(cls, 'lower'):
            cls = class_by_name[cls]

        attrs = cls._necessary_attrs + cls._recommended_attrs

        attrnames = [attr[0] for attr in attrs]
        attrtypes = [attr[1] for attr in attrs]

        attritems = zip(attrnames, attrtypes)

        attrannnames = attrnames + list(self.annotations.keys())

        self.assertEqual(sorted(attrnames), sorted(res1.keys()))
        self.assertEqual(sorted(attrannnames), sorted(res2.keys()))

        items11 = [(name, type(value)) for name, value in res1.items()]
        self.assertEqual(sorted(attritems), sorted(items11))
        for name, value in res1.items():
            try:
                self.assertEqual(res2[name], value)
            except ValueError:
                assert_arrays_equal(res2[name], value)

        for name, value in self.annotations.items():
            self.assertFalse(name in res1)
            self.assertEqual(res2[name], value)

        for attr in attrs:
            name = attr[0]
            if len(attr) < 3:
                continue

            dim = attr[2]
            self.assertEqual(dim, res1[name].ndim)
            self.assertEqual(dim, res2[name].ndim)

            if len(attr) < 4:
                continue

            dtype = attr[3]
            self.assertEqual(dtype.kind, res1[name].dtype.kind)
            self.assertEqual(dtype.kind, res2[name].dtype.kind)

    def check__get_fake_values(self, cls):
        self.subcheck__get_fake_values(cls)
        self.subcheck__get_fake_values(cls.__name__)

    def test__analogsignal(self):
        self.check__get_fake_values(AnalogSignal)

    def test__analogsignalarray(self):
        self.check__get_fake_values(AnalogSignalArray)

    def test__block(self):
        self.check__get_fake_values(Block)

    def test__epoch(self):
        self.check__get_fake_values(Epoch)

    def test__epocharray(self):
        self.check__get_fake_values(EpochArray)

    def test__event(self):
        self.check__get_fake_values(Event)

    def test__eventarray(self):
        self.check__get_fake_values(EventArray)

    def test__irregularlysampledsignal(self):
        self.check__get_fake_values(IrregularlySampledSignal)

    def test__recordingchannel(self):
        self.check__get_fake_values(RecordingChannel)

    def test__recordingchannelgroup(self):
        self.check__get_fake_values(RecordingChannelGroup)

    def test__segment(self):
        self.check__get_fake_values(Segment)

    def test__spike(self):
        self.check__get_fake_values(Spike)

    def test__spiketrain(self):
        self.check__get_fake_values(SpikeTrain)

    def test__unit(self):
        self.check__get_fake_values(Unit)


class Test__generate_datasets(unittest.TestCase):
    def setUp(self):
        self.annotations = dict([(str(x), TEST_ANNOTATIONS[x]) for x in
                                 range(len(TEST_ANNOTATIONS))])

    def check__generate_datasets(self, cls):
        clsname = cls.__name__

        self.subcheck__generate_datasets(cls, cascade=True)
        self.subcheck__generate_datasets(cls, cascade=True, seed=0)
        self.subcheck__generate_datasets(cls, cascade=False)
        self.subcheck__generate_datasets(cls, cascade=False, seed=0)
        self.subcheck__generate_datasets(clsname, cascade=True)
        self.subcheck__generate_datasets(clsname, cascade=True, seed=0)
        self.subcheck__generate_datasets(clsname, cascade=False)
        self.subcheck__generate_datasets(clsname, cascade=False, seed=0)

    def subcheck__generate_datasets(self, cls, cascade, seed=None):
        self.annotations['seed'] = seed

        if seed is None:
            res = fake_neo(obj_type=cls, cascade=cascade)
        else:
            res = fake_neo(obj_type=cls, cascade=cascade, seed=seed)

        if not hasattr(cls, 'lower'):
            self.assertTrue(isinstance(res, cls))
        else:
            self.assertEqual(res.__class__.__name__, cls)

        assert_neo_object_is_compliant(res)
        self.assertEqual(res.annotations, self.annotations)

        resattr = get_fake_values(cls, annotate=False, seed=0)
        if seed is not None:
            for name, value in resattr.items():
                if name in ['channel_names',
                            'channel_indexes',
                            'channel_index']:
                    continue
                try:
                    try:
                        resvalue = getattr(res, name)
                    except AttributeError:
                        if name == 'signal':
                            continue
                        raise
                    try:
                        self.assertEqual(resvalue, value)
                    except ValueError:
                        assert_arrays_equal(resvalue, value)
                except BaseException as exc:
                    exc.args += ('from %s' % name,)
                    raise

        if not getattr(res, '_child_objects', ()):
            pass
        elif not cascade:
            self.assertEqual(res.children, ())
        else:
            self.assertNotEqual(res.children, ())

        if cls in ['RecordingChannelGroup', RecordingChannelGroup]:
            for i, rchan in enumerate(res.recordingchannels):
                self.assertEqual(rchan.name, res.channel_names[i].astype(str))
                self.assertEqual(rchan.index, res.channel_indexes[i])
            for i, unit in enumerate(res.units):
                for sigarr in res.analogsignalarrays:
                    self.assertEqual(unit.channel_indexes[0],
                                     sigarr.channel_index[i])

    def test__analogsignal(self):
        self.check__generate_datasets(AnalogSignal)

    def test__analogsignalarray(self):
        self.check__generate_datasets(AnalogSignal)

    def test__block(self):
        self.check__generate_datasets(AnalogSignalArray)

    def test__epoch(self):
        self.check__generate_datasets(Epoch)

    def test__epocharray(self):
        self.check__generate_datasets(EpochArray)

    def test__event(self):
        self.check__generate_datasets(Event)

    def test__eventarray(self):
        self.check__generate_datasets(EventArray)

    def test__irregularlysampledsignal(self):
        self.check__generate_datasets(IrregularlySampledSignal)

    def test__recordingchannel(self):
        self.check__generate_datasets(RecordingChannel)

    def test__recordingchannelgroup(self):
        self.check__generate_datasets(RecordingChannelGroup)

    def test__segment(self):
        self.check__generate_datasets(Segment)

    def test__spike(self):
        self.check__generate_datasets(Spike)

    def test__spiketrain(self):
        self.check__generate_datasets(SpikeTrain)

    def test__unit(self):
        self.check__generate_datasets(Unit)
