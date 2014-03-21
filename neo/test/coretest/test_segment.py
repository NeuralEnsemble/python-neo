# -*- coding: utf-8 -*-
"""
Tests of the neo.core.segment.Segment class
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

from datetime import datetime

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from neo.core.segment import Segment
from neo.core import (AnalogSignal, AnalogSignalArray, Block,
                      Epoch, EpochArray, Event, EventArray,
                      IrregularlySampledSignal, RecordingChannelGroup,
                      Spike, SpikeTrain, Unit)
from neo.test.tools import (assert_neo_object_is_compliant,
                            assert_arrays_equal,
                            assert_same_sub_schema)
from neo.test.generate_datasets import (fake_neo, get_fake_value,
                                        get_fake_values, get_annotations,
                                        clone_object, TEST_ANNOTATIONS)


class Test__generate_datasets(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.annotations = dict([(str(x), TEST_ANNOTATIONS[x]) for x in
                                 range(len(TEST_ANNOTATIONS))])

    def test__get_fake_values(self):
        self.annotations['seed'] = 0
        file_datetime = get_fake_value('file_datetime', datetime, seed=0)
        rec_datetime = get_fake_value('rec_datetime', datetime, seed=1)
        index = get_fake_value('index', int, seed=2)
        name = get_fake_value('name', str, seed=3, obj=Segment)
        description = get_fake_value('description', str, seed=4, obj='Segment')
        file_origin = get_fake_value('file_origin', str)
        attrs1 = {'file_datetime': file_datetime,
                  'rec_datetime': rec_datetime,
                  'index': index,
                  'name': name,
                  'description': description,
                  'file_origin': file_origin}
        attrs2 = attrs1.copy()
        attrs2.update(self.annotations)

        res11 = get_fake_values(Segment, annotate=False, seed=0)
        res12 = get_fake_values('Segment', annotate=False, seed=0)
        res21 = get_fake_values(Segment, annotate=True, seed=0)
        res22 = get_fake_values('Segment', annotate=True, seed=0)

        self.assertEqual(res11, attrs1)
        self.assertEqual(res12, attrs1)
        self.assertEqual(res21, attrs2)
        self.assertEqual(res22, attrs2)

    def test__fake_neo__cascade(self):
        self.annotations['seed'] = None
        obj_type = Segment
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
        for child in res.children:
            del child.annotations['i']
            del child.annotations['j']

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

    def test__fake_neo__nocascade(self):
        self.annotations['seed'] = None
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


class TestSegment(unittest.TestCase):
    def setUp(self):
        self.nchildren = 2
        blk = fake_neo(Block, seed=0, n=self.nchildren)
        self.unit1, self.unit2, self.unit3, self.unit4 = blk.list_units
        self.seg1, self.seg2 = blk.segments
        self.targobj = self.seg1
        self.seed1 = self.seg1.annotations['seed']
        self.seed2 = self.seg2.annotations['seed']

        del self.seg1.annotations['i']
        del self.seg2.annotations['i']
        del self.seg1.annotations['j']
        del self.seg2.annotations['j']

        self.sigs1 = self.seg1.analogsignals
        self.sigs2 = self.seg2.analogsignals
        self.sigarrs1 = self.seg1.analogsignalarrays
        self.sigarrs2 = self.seg2.analogsignalarrays
        self.irsigs1 = self.seg1.irregularlysampledsignals
        self.irsigs2 = self.seg2.irregularlysampledsignals

        self.spikes1 = self.seg1.spikes
        self.spikes2 = self.seg2.spikes
        self.trains1 = self.seg1.spiketrains
        self.trains2 = self.seg2.spiketrains

        self.epcs1 = self.seg1.epochs
        self.epcs2 = self.seg2.epochs
        self.epcas1 = self.seg1.epocharrays
        self.epcas2 = self.seg2.epocharrays
        self.evts1 = self.seg1.events
        self.evts2 = self.seg2.events
        self.evtas1 = self.seg1.eventarrays
        self.evtas2 = self.seg2.eventarrays

        self.sigs1a = clone_object(self.sigs1)
        self.sigarrs1a = clone_object(self.sigarrs1, n=2)
        self.irsigs1a = clone_object(self.irsigs1)

        self.spikes1a = clone_object(self.spikes1)
        self.trains1a = clone_object(self.trains1)

        self.epcs1a = clone_object(self.epcs1)
        self.epcas1a = clone_object(self.epcas1)
        self.evts1a = clone_object(self.evts1)
        self.evtas1a = clone_object(self.evtas1)

        for obj, obja in zip(self.sigs1 + self.sigarrs1,
                             self.sigs1a + self.sigarrs1a):
            obja.channel_index = obj.channel_index

    def test_init(self):
        seg = Segment(name='a segment', index=3)
        assert_neo_object_is_compliant(seg)
        self.assertEqual(seg.name, 'a segment')
        self.assertEqual(seg.file_origin, None)
        self.assertEqual(seg.index, 3)

    def check_creation(self, seg):
        assert_neo_object_is_compliant(seg)

        seed = seg.annotations['seed']

        targ0 = get_fake_value('file_datetime', datetime, seed=seed+0)
        self.assertEqual(seg.file_datetime, targ0)

        targ1 = get_fake_value('rec_datetime', datetime, seed=seed+1)
        self.assertEqual(seg.rec_datetime, targ1)

        targ2 = get_fake_value('index', int, seed=seed+2)
        self.assertEqual(seg.index, targ2)

        targ3 = get_fake_value('name', str, seed=seed+3, obj=Segment)
        self.assertEqual(seg.name, targ3)

        targ4 = get_fake_value('description', str,
                               seed=seed+4, obj=Segment)
        self.assertEqual(seg.description, targ4)

        targ5 = get_fake_value('file_origin', str)
        self.assertEqual(seg.file_origin, targ5)

        targ6 = get_annotations()
        targ6['seed'] = seed
        self.assertEqual(seg.annotations, targ6)

        self.assertTrue(hasattr(seg, 'analogsignals'))
        self.assertTrue(hasattr(seg, 'analogsignalarrays'))
        self.assertTrue(hasattr(seg, 'irregularlysampledsignals'))

        self.assertTrue(hasattr(seg, 'epochs'))
        self.assertTrue(hasattr(seg, 'epocharrays'))
        self.assertTrue(hasattr(seg, 'events'))
        self.assertTrue(hasattr(seg, 'eventarrays'))

        self.assertTrue(hasattr(seg, 'spikes'))
        self.assertTrue(hasattr(seg, 'spiketrains'))

        self.assertEqual(len(seg.analogsignals), self.nchildren**2)
        self.assertEqual(len(seg.analogsignalarrays), self.nchildren)
        self.assertEqual(len(seg.irregularlysampledsignals), self.nchildren**2)

        self.assertEqual(len(seg.epochs), self.nchildren)
        self.assertEqual(len(seg.epocharrays), self.nchildren)
        self.assertEqual(len(seg.events), self.nchildren)
        self.assertEqual(len(seg.eventarrays), self.nchildren)

        self.assertEqual(len(seg.spikes), self.nchildren**2)
        self.assertEqual(len(seg.spiketrains), self.nchildren**2)

    def test__creation(self):
        self.check_creation(self.seg1)
        self.check_creation(self.seg2)

    def test__merge(self):
        seg1a = fake_neo(Block, seed=self.seed1, n=self.nchildren).segments[0]
        assert_same_sub_schema(self.seg1, seg1a)
        seg1a.spikes.append(self.spikes2[0])
        seg1a.epocharrays.append(self.epcas2[0])
        seg1a.annotate(seed=self.seed2)
        seg1a.merge(self.seg2)
        self.check_creation(self.seg2)
        self.epcas2[0] = self.epcas2[0].merge(self.epcas2[0])

        assert_same_sub_schema(self.sigs1a + self.sigs2, seg1a.analogsignals)
        assert_same_sub_schema(self.sigarrs1a + self.sigarrs2,
                               seg1a.analogsignalarrays)
        assert_same_sub_schema(self.irsigs1a + self.irsigs2,
                               seg1a.irregularlysampledsignals)

        assert_same_sub_schema(self.epcs1 + self.epcs2, seg1a.epochs)
        assert_same_sub_schema(self.epcas1 + self.epcas2, seg1a.epocharrays)
        assert_same_sub_schema(self.evts1 + self.evts2, seg1a.events)
        assert_same_sub_schema(self.evtas1 + self.evtas2, seg1a.eventarrays)

        assert_same_sub_schema(self.spikes1 + self.spikes2[:1] + self.spikes2,
                               seg1a.spikes)
        assert_same_sub_schema(self.trains1 + self.trains2, seg1a.spiketrains)

    def test__children(self):
        blk = Block(name='block1')
        blk.segments = [self.seg1]
        blk.create_many_to_one_relationship(force=True)
        assert_neo_object_is_compliant(self.seg1)
        assert_neo_object_is_compliant(blk)

        childobjs = ('AnalogSignal', 'AnalogSignalArray',
                     'Epoch', 'EpochArray',
                     'Event', 'EventArray',
                     'IrregularlySampledSignal',
                     'Spike', 'SpikeTrain')
        childconts = ('analogsignals', 'analogsignalarrays',
                      'epochs', 'epocharrays',
                      'events', 'eventarrays',
                      'irregularlysampledsignals',
                      'spikes', 'spiketrains')
        self.assertEqual(self.seg1._container_child_objects, ())
        self.assertEqual(self.seg1._data_child_objects, childobjs)
        self.assertEqual(self.seg1._single_parent_objects, ('Block',))
        self.assertEqual(self.seg1._multi_child_objects, ())
        self.assertEqual(self.seg1._multi_parent_objects, ())
        self.assertEqual(self.seg1._child_properties, ())

        self.assertEqual(self.seg1._single_child_objects, childobjs)
        self.assertEqual(self.seg1._container_child_containers, ())
        self.assertEqual(self.seg1._data_child_containers, childconts)
        self.assertEqual(self.seg1._single_child_containers, childconts)
        self.assertEqual(self.seg1._single_parent_containers, ('block',))
        self.assertEqual(self.seg1._multi_child_containers, ())
        self.assertEqual(self.seg1._multi_parent_containers, ())

        self.assertEqual(self.seg1._child_objects, childobjs)
        self.assertEqual(self.seg1._child_containers, childconts)
        self.assertEqual(self.seg1._parent_objects, ('Block',))
        self.assertEqual(self.seg1._parent_containers, ('block',))

        totchildren = (self.nchildren*2*2 +  # epoch/event(array)
                       self.nchildren +  # analogsignalarray
                       2*(self.nchildren**2) +  # spike(train)
                       2*(self.nchildren**2))  # analog/irregsignal
        self.assertEqual(len(self.seg1.children), totchildren)

        children = (self.sigs1a + self.sigarrs1a +
                    self.epcs1a + self.epcas1a +
                    self.evts1a + self.evtas1a +
                    self.irsigs1a +
                    self.spikes1a + self.trains1a)
        assert_same_sub_schema(list(self.seg1.children), children)

        self.assertEqual(len(self.seg1.parents), 1)
        self.assertEqual(self.seg1.parents[0].name, 'block1')

    def test__construct_subsegment_by_unit(self):
        nb_seg = 3
        nb_unit = 7
        unit_with_sig = np.array([0, 2, 5])
        signal_types = ['Vm', 'Conductances']
        sig_len = 100

        #recordingchannelgroups
        rcgs = [RecordingChannelGroup(name='Vm',
                                      channel_indexes=unit_with_sig),
                RecordingChannelGroup(name='Conductance',
                                      channel_indexes=unit_with_sig)]

        # Unit
        all_unit = []
        for u in range(nb_unit):
            un = Unit(name='Unit #%d' % u, channel_indexes=np.array([u]))
            assert_neo_object_is_compliant(un)
            all_unit.append(un)

        blk = Block()
        blk.recordingchannelgroups = rcgs
        for s in range(nb_seg):
            seg = Segment(name='Simulation %s' % s)
            for j in range(nb_unit):
                st = SpikeTrain([1, 2], units='ms',
                                t_start=0., t_stop=10)
                st.unit = all_unit[j]

            for t in signal_types:
                anasigarr = AnalogSignalArray(np.zeros((sig_len,
                                                        len(unit_with_sig))),
                                              units='nA',
                                              sampling_rate=1000.*pq.Hz,
                                              channel_indexes=unit_with_sig)
                seg.analogsignalarrays.append(anasigarr)

        blk.create_many_to_one_relationship()
        for unit in all_unit:
            assert_neo_object_is_compliant(unit)
        for rcg in rcgs:
            assert_neo_object_is_compliant(rcg)
        assert_neo_object_is_compliant(blk)

        # what you want
        newseg = seg.construct_subsegment_by_unit(all_unit[:4])
        assert_neo_object_is_compliant(newseg)

    def test_segment_take_spikes_by_unit(self):
        result1 = self.seg1.take_spikes_by_unit()
        result21 = self.seg1.take_spikes_by_unit([self.unit1])
        result22 = self.seg1.take_spikes_by_unit([self.unit2])

        self.assertEqual(result1, [])

        assert_same_sub_schema(result21, [self.spikes1a[0]])
        assert_same_sub_schema(result22, [self.spikes1a[1]])

    def test_segment_take_spiketrains_by_unit(self):
        result1 = self.seg1.take_spiketrains_by_unit()
        result21 = self.seg1.take_spiketrains_by_unit([self.unit1])
        result22 = self.seg1.take_spiketrains_by_unit([self.unit2])

        self.assertEqual(result1, [])

        assert_same_sub_schema(result21, [self.trains1a[0]])
        assert_same_sub_schema(result22, [self.trains1a[1]])

    def test_segment_take_analogsignal_by_unit(self):
        result1 = self.seg1.take_analogsignal_by_unit()
        result21 = self.seg1.take_analogsignal_by_unit([self.unit1])
        result22 = self.seg1.take_analogsignal_by_unit([self.unit2])

        self.assertEqual(result1, [])

        assert_same_sub_schema(result21, [self.sigs1a[0]])
        assert_same_sub_schema(result22, [self.sigs1a[1]])

    def test_segment_take_analogsignal_by_channelindex(self):
        ind1 = self.unit1.channel_indexes[0]
        ind2 = self.unit2.channel_indexes[0]
        result1 = self.seg1.take_analogsignal_by_channelindex()
        result21 = self.seg1.take_analogsignal_by_channelindex([ind1])
        result22 = self.seg1.take_analogsignal_by_channelindex([ind2])

        self.assertEqual(result1, [])

        assert_same_sub_schema(result21, [self.sigs1a[0]])
        assert_same_sub_schema(result22, [self.sigs1a[1]])

    def test_seg_take_slice_of_analogsignalarray_by_unit(self):
        seg = self.seg1
        result1 = seg.take_slice_of_analogsignalarray_by_unit()
        result21 = seg.take_slice_of_analogsignalarray_by_unit([self.unit1])
        result23 = seg.take_slice_of_analogsignalarray_by_unit([self.unit3])

        self.assertEqual(result1, [])

        targ1 = [self.sigarrs1a[0][:, np.array([True])],
                 self.sigarrs1a[1][:, np.array([False])]]
        targ3 = [self.sigarrs1a[0][:, np.array([False])],
                 self.sigarrs1a[1][:, np.array([True])]]
        assert_same_sub_schema(result21, targ1)
        assert_same_sub_schema(result23, targ3)

    def test_seg_take_slice_of_analogsignalarray_by_channelindex(self):
        seg = self.seg1
        ind1 = self.unit1.channel_indexes[0]
        ind3 = self.unit3.channel_indexes[0]
        result1 = seg.take_slice_of_analogsignalarray_by_channelindex()
        result21 = seg.take_slice_of_analogsignalarray_by_channelindex([ind1])
        result23 = seg.take_slice_of_analogsignalarray_by_channelindex([ind3])

        self.assertEqual(result1, [])

        targ1 = [self.sigarrs1a[0][:, np.array([True])],
                 self.sigarrs1a[1][:, np.array([False])]]
        targ3 = [self.sigarrs1a[0][:, np.array([False])],
                 self.sigarrs1a[1][:, np.array([True])]]
        assert_same_sub_schema(result21, targ1)
        assert_same_sub_schema(result23, targ3)


if __name__ == "__main__":
    unittest.main()
