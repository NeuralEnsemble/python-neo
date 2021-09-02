"""
Tests of the neo.core.segment.Segment class
"""

from copy import deepcopy

from datetime import datetime

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
from neo.core import (AnalogSignal, Block, Event, IrregularlySampledSignal,
                      Epoch, SpikeTrain)
from neo.core.spiketrainlist import SpikeTrainList
from neo.core.container import filterdata
from neo.test.tools import (assert_neo_object_is_compliant,
                            assert_same_sub_schema, assert_same_attributes)
from neo.test.generate_datasets import random_segment, simple_block
from neo.rawio.examplerawio import ExampleRawIO
from neo.io.proxyobjects import (AnalogSignalProxy, SpikeTrainProxy,
                                 EventProxy, EpochProxy)


N_EXAMPLES = 5


class TestSegment(unittest.TestCase):

    def setUp(self):
        self.segments = [random_segment() for i in range(N_EXAMPLES)]

    def test_init(self):
        seg = Segment(name='a segment', index=3)
        assert_neo_object_is_compliant(seg)
        self.assertEqual(seg.name, 'a segment')
        self.assertEqual(seg.file_origin, None)
        self.assertEqual(seg.index, 3)

    def test_times(self):
        for seg in self.segments:
            # calculate target values for t_start and t_stop
            t_starts, t_stops = [], []
            for children in [seg.analogsignals,
                             seg.epochs,
                             seg.events,
                             seg.irregularlysampledsignals,
                             seg.spiketrains]:
                for child in children:
                    if hasattr(child, 't_start'):
                        t_starts.append(child.t_start)
                    if hasattr(child, 't_stop'):
                        t_stops.append(child.t_stop)
                    if hasattr(child, 'time'):
                        t_starts.append(child.time)
                        t_stops.append(child.time)
                    if hasattr(child, 'times'):
                        t_starts.append(child.times[0])
                        t_stops.append(child.times[-1])
            targ_t_start = min(t_starts)
            targ_t_stop = max(t_stops)

            self.assertEqual(seg.t_start, targ_t_start)
            self.assertEqual(seg.t_stop, targ_t_stop)

        # Testing times with ProxyObjects
        seg = Segment()
        reader = ExampleRawIO(filename='my_filename.fake')
        reader.parse_header()

        proxy_anasig = AnalogSignalProxy(rawio=reader,
                        stream_index=0, inner_stream_channels=None,
                        block_index=0, seg_index=0)
        seg.analogsignals.append(proxy_anasig)

        proxy_st = SpikeTrainProxy(rawio=reader, spike_channel_index=0, block_index=0, seg_index=0)
        seg.spiketrains.append(proxy_st)

        proxy_event = EventProxy(rawio=reader, event_channel_index=0, block_index=0, seg_index=0)
        seg.events.append(proxy_event)

        proxy_epoch = EpochProxy(rawio=reader, event_channel_index=1, block_index=0, seg_index=0)
        seg.epochs.append(proxy_epoch)

        t_starts, t_stops = [], []
        for children in [seg.analogsignals,
                         seg.epochs,
                         seg.events,
                         seg.irregularlysampledsignals,
                         seg.spiketrains]:
            for child in children:
                if hasattr(child, 't_start'):
                    t_starts.append(child.t_start)
                if hasattr(child, 't_stop'):
                    t_stops.append(child.t_stop)
                if hasattr(child, 'time'):
                    t_starts.append(child.time)
                    t_stops.append(child.time)
                if hasattr(child, 'times'):
                    t_starts.append(child.times[0])
                    t_stops.append(child.times[-1])
        targ_t_start = min(t_starts)
        targ_t_stop = max(t_stops)

        self.assertEqual(seg.t_start, targ_t_start)
        self.assertEqual(seg.t_stop, targ_t_stop)

    def test__merge(self):
        seg1 = self.segments[0]
        seg2 = self.segments[1]
        orig_seg1 = deepcopy(seg1)
        seg1.merge(seg2)

        assert_same_sub_schema(orig_seg1.analogsignals + seg2.analogsignals,
                               seg1.analogsignals)
        assert_same_sub_schema(
            orig_seg1.irregularlysampledsignals + seg2.irregularlysampledsignals,
            seg1.irregularlysampledsignals)
        assert_same_sub_schema(orig_seg1.epochs + seg2.epochs, seg1.epochs)
        assert_same_sub_schema(orig_seg1.events + seg2.events, seg1.events)
        assert_same_sub_schema(orig_seg1.spiketrains + seg2.spiketrains, seg1.spiketrains)

    def test__size(self):
        for segment in self.segments:
            targ1 = {
                "epochs": len(segment.epochs),
                "events": len(segment.events),
                "irregularlysampledsignals": len(segment.irregularlysampledsignals),
                "spiketrains": len(segment.spiketrains),
                "analogsignals": len(segment.analogsignals),
                "imagesequences": len(segment.imagesequences)
            }
            self.assertEqual(segment.size, targ1)

    def test__filter_none(self):
        for segment in self.segments:
            targ = []
            # collecting all data objects in target block
            targ.extend(segment.analogsignals)
            targ.extend(segment.epochs)
            targ.extend(segment.events)
            targ.extend(segment.irregularlysampledsignals)
            targ.extend(segment.spiketrains)
            targ.extend(segment.imagesequences)

            # occasionally we randomly get only spike trains,
            # and then we have to convert to a SpikeTrainList
            # to match the output of segment.filter
            if all(isinstance(obj, SpikeTrain) for obj in targ):
                targ = SpikeTrainList(items=targ, segment=segment)

            res0 = segment.filter()
            res1 = segment.filter({})
            res2 = segment.filter([])
            res3 = segment.filter([{}])
            res4 = segment.filter([{}, {}])
            res5 = segment.filter([{}, {}])
            res6 = segment.filter(targdict={})
            res7 = segment.filter(targdict=[])
            res8 = segment.filter(targdict=[{}])
            res9 = segment.filter(targdict=[{}, {}])

            assert_same_sub_schema(res0, targ)
            assert_same_sub_schema(res1, targ)
            assert_same_sub_schema(res2, targ)
            assert_same_sub_schema(res3, targ)
            assert_same_sub_schema(res4, targ)
            assert_same_sub_schema(res5, targ)
            assert_same_sub_schema(res6, targ)
            assert_same_sub_schema(res7, targ)
            assert_same_sub_schema(res8, targ)
            assert_same_sub_schema(res9, targ)

    def test__filter_annotation_single(self):
        segment = simple_block().segments[0]
        targ = [segment.analogsignals[0], segment.spiketrains[1]]

        res0 = segment.filter(thing="wotsit")
        res1 = segment.filter({'thing': "wotsit"})
        res2 = segment.filter(targdict={'thing': "wotsit"})
        res3 = segment.filter([{'thing': "wotsit"}])
        res4 = segment.filter(targdict=[{'thing': "wotsit"}])

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)
        assert_same_sub_schema(res4, targ)

    def test__filter_single_annotation_nores(self):
        segment = simple_block().segments[0]
        targ = []

        res0 = segment.filter(j=5)
        res1 = segment.filter({'j': 5})
        res2 = segment.filter(targdict={'j': 5})
        res3 = segment.filter([{'j': 5}])
        res4 = segment.filter(targdict=[{'j': 5}])

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)
        assert_same_sub_schema(res4, targ)

    def test__filter_attribute_single(self):
        segment = simple_block().segments[1]

        targ = [segment.analogsignals[0], segment.irregularlysampledsignals[0]]

        name = targ[0].name
        res0 = segment.filter(name=name)
        res1 = segment.filter({'name': name})
        res2 = segment.filter(targdict={'name': name})

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_attribute_single_nores(self):
        segment = simple_block().segments[1]

        name = "potato"
        res0 = segment.filter(name=name)
        res1 = segment.filter({'name': name})
        res2 = segment.filter(targdict={'name': name})

        self.assertEqual(len(res0), 0)
        self.assertEqual(len(res1), 0)
        self.assertEqual(len(res2), 0)

    def test__filter_multi(self):

        segment = simple_block().segments[1]
        targ = [segment.analogsignals[0], segment.irregularlysampledsignals[0]]

        filter = {
            "name": targ[0].name,
            "thing": targ[0].annotations["thing"]
        }

        res0 = segment.filter(**filter)
        res1 = segment.filter(filter)
        res2 = segment.filter(targdict=filter)

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_multi_nores(self):
        segment = simple_block().segments[0]
        targ = []

        filter = {
            "name": "carrot",
            "thing": "another thing"
        }

        res0 = segment.filter(**filter)
        res1 = segment.filter(filter)
        res2 = segment.filter(targdict=filter)

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_multi_partres(self):
        segment = simple_block().segments[1]
        targ = [segment.analogsignals[0], segment.irregularlysampledsignals[0]]

        filter = {
            "name": targ[0].name,
            "thing": "another thing"
        }

        res0 = segment.filter(**filter)
        res1 = segment.filter(filter)
        res2 = segment.filter(targdict=filter)

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_no_annotation_but_object(self):
        for segment in self.segments:
            targ = segment.spiketrains
            assert isinstance(targ, SpikeTrainList)
            res = segment.filter(objects=SpikeTrain)
            if len(res) > 0:
                # if res has length 0 it will be just a plain list
                assert isinstance(res, SpikeTrainList)
                assert_same_sub_schema(res, targ)

            targ = segment.analogsignals
            res = segment.filter(objects=AnalogSignal)
            assert_same_sub_schema(res, targ)

            targ = segment.analogsignals + segment.spiketrains
            res = segment.filter(objects=[AnalogSignal, SpikeTrain])
            if len(res) > 0:
                assert_same_sub_schema(res, targ)

    def test__filter_single_annotation_obj_single(self):
        segment = simple_block().segments[0]
        targ = [segment.analogsignals[1]]

        res0 = segment.filter(thing="frooble", objects='AnalogSignal')
        res1 = segment.filter(thing="frooble", objects=AnalogSignal)
        res2 = segment.filter(thing="frooble", objects=['AnalogSignal'])
        res3 = segment.filter(thing="frooble", objects=[AnalogSignal])

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)
        assert_same_sub_schema(res3, targ)

    def test__filter_single_annotation_obj_multi(self):
        segment = simple_block().segments[0]
        targ = [segment.analogsignals[0], segment.spiketrains[1]]

        res0 = segment.filter(thing="wotsit",
                              objects=["Event", SpikeTrain, "AnalogSignal"])

        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_obj_none(self):
        segment = simple_block().segments[0]
        targ = []

        res0 = segment.filter(j=1, objects=Epoch)
        res1 = segment.filter(j=1, objects='Epoch')
        res2 = segment.filter(j=1, objects=[])

        assert_same_sub_schema(res0, targ)
        assert_same_sub_schema(res1, targ)
        assert_same_sub_schema(res2, targ)

    def test__filter_single_annotation_nodata(self):
        segment = simple_block().segments[0]
        targ = []
        res0 = segment.filter(thing="frooble", data=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata(self):
        segment = simple_block().segments[0]
        targ = []
        res0 = segment.filter(name=segment.analogsignals[0], data=False)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_container(self):
        segment = simple_block().segments[1]

        targ = [
            segment.analogsignals[0],
        ]

        res0 = segment.filter(thing="amajig", container=True)

        self.assertEqual(res0, targ)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_container(self):
        segment = simple_block().segments[1]

        targ = [segment.analogsignals[0], segment.irregularlysampledsignals[0]]
        res0 = segment.filter(name=targ[0].name, container=True)

        self.assertEqual(res0, targ)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_annotation_container(self):
        segment = simple_block().segments[1]

        targ = []
        res0 = segment.filter(thing="amajig", container=True, data=False)

        self.assertEqual(res0, targ)
        assert_same_sub_schema(res0, targ)

    def test__filter_single_attribute_nodata_container(self):
        segment = simple_block().segments[1]

        targ = []
        res0 = segment.filter(name=segment.analogsignals[0].name, container=True, data=False)

        self.assertEqual(res0, targ)
        assert_same_sub_schema(res0, targ)

    # @unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
    # def test__pretty(self):
    #     ann = get_annotations()
    #     ann['seed'] = self.seed1
    #     ann = pretty(ann).replace('\n ', '\n  ')
    #     res = pretty(self.seg1)
    #
    #     sigarr0 = pretty(self.sigarrs1[0])
    #     sigarr1 = pretty(self.sigarrs1[1])
    #     sigarr0 = sigarr0.replace('\n', '\n   ')
    #     sigarr1 = sigarr1.replace('\n', '\n   ')
    #
    #     targ = ("Segment with " +
    #             ("%s analogsignals, " %
    #              (len(self.sigarrs1a),)) +
    #             ("%s epochs, " % len(self.epcs1a)) +
    #             ("%s events, " % len(self.evts1a)) +
    #             ("%s irregularlysampledsignals, " %
    #              len(self.irsigs1a)) +
    #             ("%s spiketrains\n" % len(self.trains1a)) +
    #             ("name: '%s'\ndescription: '%s'\n" %
    #              (self.seg1.name, self.seg1.description)
    #              ) +
    #
    #             ("annotations: %s\n" % ann) +
    #
    #             ("# analogsignals (N=%s)\n" % len(self.sigarrs1a)) +
    #
    #             ('%s: %s\n' % (0, sigarr0)) +
    #             ('%s: %s' % (1, sigarr1)))
    #
    #     self.assertEqual(res, targ)

    def test__time_slice(self):
        time_slice = [.5, 5.6] * pq.s

        epoch2 = Epoch([0.6, 9.5, 16.8, 34.1] * pq.s, durations=[4.5, 4.8, 5.0, 5.0] * pq.s,
                       t_start=.1 * pq.s)
        epoch2.annotate(epoch_type='b')
        epoch2.array_annotate(trial_id=[1, 2, 3, 4])

        event = Event(times=[0.5, 10.0, 25.2] * pq.s, t_start=.1 * pq.s)
        event.annotate(event_type='trial start')
        event.array_annotate(trial_id=[1, 2, 3])

        anasig = AnalogSignal(np.arange(50.0) * pq.mV, t_start=.1 * pq.s,
                              sampling_rate=1.0 * pq.Hz)
        irrsig = IrregularlySampledSignal(signal=np.arange(50.0) * pq.mV,
                                          times=anasig.times, t_start=.1 * pq.s)
        st = SpikeTrain(np.arange(0.5, 50, 7) * pq.s, t_start=.1 * pq.s, t_stop=50.0 * pq.s,
                        waveforms=np.array([[[0., 1.], [0.1, 1.1]], [[2., 3.], [2.1, 3.1]],
                                            [[4., 5.], [4.1, 5.1]], [[6., 7.], [6.1, 7.1]],
                                            [[8., 9.], [8.1, 9.1]], [[12., 13.], [12.1, 13.1]],
                                            [[14., 15.], [14.1, 15.1]],
                                            [[16., 17.], [16.1, 17.1]]]) * pq.mV,
                        array_annotations={'spikenum': np.arange(1, 9)})

        seg = Segment()
        seg.epochs = [epoch2]
        seg.events = [event]
        seg.analogsignals = [anasig]
        seg.irregularlysampledsignals = [irrsig]
        seg.spiketrains = [st]

        block = Block()
        block.segments = [seg]
        block.create_many_to_one_relationship()

        # test without resetting the time
        sliced = seg.time_slice(time_slice[0], time_slice[1])

        assert_neo_object_is_compliant(sliced)

        self.assertEqual(len(sliced.events), 1)
        self.assertEqual(len(sliced.spiketrains), 1)
        self.assertEqual(len(sliced.analogsignals), 1)
        self.assertEqual(len(sliced.irregularlysampledsignals), 1)
        self.assertEqual(len(sliced.epochs), 1)

        assert_same_attributes(sliced.spiketrains[0],
                               st.time_slice(t_start=time_slice[0],
                                             t_stop=time_slice[1]))
        assert_same_attributes(sliced.analogsignals[0],
                               anasig.time_slice(t_start=time_slice[0],
                                                 t_stop=time_slice[1]))
        assert_same_attributes(sliced.irregularlysampledsignals[0],
                               irrsig.time_slice(t_start=time_slice[0],
                                                 t_stop=time_slice[1]))
        assert_same_attributes(sliced.events[0],
                               event.time_slice(t_start=time_slice[0],
                                                t_stop=time_slice[1]))
        assert_same_attributes(sliced.epochs[0],
                               epoch2.time_slice(t_start=time_slice[0],
                                                 t_stop=time_slice[1]))

        seg = Segment()
        seg.epochs = [epoch2]
        seg.events = [event]
        seg.analogsignals = [anasig]
        seg.irregularlysampledsignals = [irrsig]
        seg.spiketrains = [st]

        block = Block()
        block.segments = [seg]
        block.create_many_to_one_relationship()

        # test with resetting the time
        sliced = seg.time_slice(time_slice[0], time_slice[1], reset_time=True)

        assert_neo_object_is_compliant(sliced)

        self.assertEqual(len(sliced.events), 1)
        self.assertEqual(len(sliced.spiketrains), 1)
        self.assertEqual(len(sliced.analogsignals), 1)
        self.assertEqual(len(sliced.irregularlysampledsignals), 1)
        self.assertEqual(len(sliced.epochs), 1)

        assert_same_attributes(sliced.spiketrains[0],
                               st.time_shift(- time_slice[0]).time_slice(
                                   t_start=0 * pq.s, t_stop=time_slice[1] - time_slice[0]))

        anasig_target = anasig.copy()
        anasig_target = anasig_target.time_shift(- time_slice[0]).time_slice(t_start=0 * pq.s,
                                                                             t_stop=time_slice[1] - time_slice[0])
        assert_same_attributes(sliced.analogsignals[0], anasig_target)
        irrsig_target = irrsig.copy()
        irrsig_target = irrsig_target.time_shift(- time_slice[0]).time_slice(t_start=0 * pq.s,
                                                                             t_stop=time_slice[1] - time_slice[0])
        assert_same_attributes(sliced.irregularlysampledsignals[0], irrsig_target)
        assert_same_attributes(sliced.events[0],
                               event.time_shift(- time_slice[0]).time_slice(
                                   t_start=0 * pq.s, t_stop=time_slice[1] - time_slice[0]))
        assert_same_attributes(sliced.epochs[0],
                               epoch2.time_shift(- time_slice[0]).time_slice(t_start=0 * pq.s,
                                                                             t_stop=time_slice[1] - time_slice[0]))

        seg = Segment()

        reader = ExampleRawIO(filename='my_filename.fake')
        reader.parse_header()

        proxy_anasig = AnalogSignalProxy(rawio=reader,
                                        stream_index=0, inner_stream_channels=None,
                                        block_index=0, seg_index=0)
        seg.analogsignals.append(proxy_anasig)

        proxy_st = SpikeTrainProxy(rawio=reader, spike_channel_index=0,
                                   block_index=0, seg_index=0)
        seg.spiketrains.append(proxy_st)

        proxy_event = EventProxy(rawio=reader, event_channel_index=0,
                                 block_index=0, seg_index=0)
        seg.events.append(proxy_event)

        proxy_epoch = EpochProxy(rawio=reader, event_channel_index=1,
                                 block_index=0, seg_index=0)
        proxy_epoch.annotate(pick='me')
        seg.epochs.append(proxy_epoch)

        loaded_epoch = proxy_epoch.load()
        loaded_event = proxy_event.load()
        loaded_st = proxy_st.load()
        loaded_anasig = proxy_anasig.load()

        block = Block()
        block.segments = [seg]
        block.create_many_to_one_relationship()

        # test with proxy objects
        sliced = seg.time_slice(time_slice[0], time_slice[1])

        assert_neo_object_is_compliant(sliced)

        sliced_event = loaded_event.time_slice(t_start=time_slice[0],
                                               t_stop=time_slice[1])
        has_event = len(sliced_event) > 0

        sliced_anasig = loaded_anasig.time_slice(t_start=time_slice[0],
                                                 t_stop=time_slice[1])

        sliced_st = loaded_st.time_slice(t_start=time_slice[0],
                                         t_stop=time_slice[1])

        self.assertEqual(len(sliced.events), int(has_event))
        self.assertEqual(len(sliced.spiketrains), 1)
        self.assertEqual(len(sliced.analogsignals), 1)

        self.assertTrue(isinstance(sliced.spiketrains[0],
                                   SpikeTrain))
        assert_same_attributes(sliced.spiketrains[0],
                               sliced_st)

        self.assertTrue(isinstance(sliced.analogsignals[0],
                                   AnalogSignal))
        assert_same_attributes(sliced.analogsignals[0],
                               sliced_anasig)

        if has_event:
            self.assertTrue(isinstance(sliced.events[0],
                                       Event))
            assert_same_attributes(sliced.events[0],
                                   sliced_event)

    def test_time_slice_None(self):
        time_slices = [(None, 5.0 * pq.s), (5.0 * pq.s, None), (None, None)]

        anasig = AnalogSignal(np.arange(50.0) * pq.mV, sampling_rate=1.0 * pq.Hz)
        seg = Segment()
        seg.analogsignals = [anasig]

        block = Block()
        block.segments = [seg]
        block.create_many_to_one_relationship()

        # test without resetting the time
        for t_start, t_stop in time_slices:
            sliced = seg.time_slice(t_start, t_stop)

            assert_neo_object_is_compliant(sliced)
            self.assertEqual(len(sliced.analogsignals), 1)

            exp_t_start, exp_t_stop = t_start, t_stop
            if exp_t_start is None:
                exp_t_start = seg.t_start
            if exp_t_stop is None:
                exp_t_stop = seg.t_stop

            self.assertEqual(exp_t_start, sliced.t_start)
            self.assertEqual(exp_t_stop, sliced.t_stop)

    def test__deepcopy(self):
        childconts = ('analogsignals',
                      'epochs', 'events',
                      'irregularlysampledsignals',
                      'spiketrains')
        for segment in self.segments:
            seg1_copy = deepcopy(segment)

            # Same structure top-down, i.e. links from parents to children are correct
            assert_same_sub_schema(seg1_copy, segment)

            # Correct structure bottom-up, i.e. links from children to parents are correct
            # No need to cascade, all children are leaves, i.e. don't have any children
            for childtype in childconts:
                for child in getattr(seg1_copy, childtype, []):
                    self.assertEqual(id(child.segment), id(seg1_copy))


if __name__ == "__main__":
    unittest.main()
