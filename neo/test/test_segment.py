# -*- coding: utf-8 -*-
"""
Tests of the neo.core.segment.Segment class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

from neo.core.segment import Segment
from neo.core import (AnalogSignal, AnalogSignalArray, Block,
                      Epoch, EpochArray, Event, EventArray,
                      IrregularlySampledSignal, RecordingChannelGroup,
                      Spike, SpikeTrain, Unit)
from neo.io.tools import create_many_to_one_relationship
from neo.test.tools import assert_neo_object_is_compliant, assert_arrays_equal


class TestSegment(unittest.TestCase):
    def setUp(self):
        self.setup_analogsignals()
        self.setup_analogsignalarrays()
        self.setup_epochs()
        self.setup_epocharrays()
        self.setup_events()
        self.setup_eventarrays()
        self.setup_irregularlysampledsignals()
        self.setup_spikes()
        self.setup_spiketrains()

        self.setup_units()
        self.setup_segments()

    def setup_segments(self):
        params = {'testarg2': 'yes', 'testarg3': True}
        self.segment1 = Segment(name='test', description='tester 1',
                                file_origin='test.file',
                                testarg1=1, **params)
        self.segment2 = Segment(name='test', description='tester 2',
                                file_origin='test.file',
                                testarg1=1, **params)
        self.segment1.annotate(testarg1=1.1, testarg0=[1, 2, 3])
        self.segment2.annotate(testarg11=1.1, testarg10=[1, 2, 3])

        self.segment1.analogsignals = self.sig1
        self.segment2.analogsignals = self.sig2

        self.segment1.analogsignalarrays = self.sigarr1
        self.segment2.analogsignalarrays = self.sigarr2

        self.segment1.epochs = self.epoch1
        self.segment2.epochs = self.epoch2

        self.segment1.epocharrays = self.epocharr1
        self.segment2.epocharrays = self.epocharr2

        self.segment1.events = self.event1
        self.segment2.events = self.event2

        self.segment1.eventarrays = self.eventarr1
        self.segment2.eventarrays = self.eventarr2

        self.segment1.irregularlysampledsignals = self.irsig1
        self.segment2.irregularlysampledsignals = self.irsig2

        self.segment1.spikes = self.spike1
        self.segment2.spikes = self.spike2

        self.segment1.spiketrains = self.train1
        self.segment2.spiketrains = self.train2

        create_many_to_one_relationship(self.segment1)
        create_many_to_one_relationship(self.segment2)

    def setup_units(self):
        params = {'testarg2': 'yes', 'testarg3': True}
        self.unit1 = Unit(name='test', description='tester 1',
                          file_origin='test.file',
                          channel_indexes=np.array([1]),
                          testarg1=1, **params)
        self.unit2 = Unit(name='test', description='tester 2',
                          file_origin='test.file',
                          channel_indexes=np.array([2]),
                          testarg1=1, **params)
        self.unit1.annotate(testarg1=1.1, testarg0=[1, 2, 3])
        self.unit2.annotate(testarg11=1.1, testarg10=[1, 2, 3])

        self.unit1train = [self.train1[0], self.train2[1]]
        self.unit2train = [self.train1[1], self.train2[0]]

        self.unit1.spiketrains = self.unit1train
        self.unit2.spiketrains = self.unit2train

        self.unit1spike = [self.spike1[0], self.spike2[1]]
        self.unit2spike = [self.spike1[1], self.spike2[0]]

        self.unit1.spikes = self.unit1spike
        self.unit2.spikes = self.unit2spike

        create_many_to_one_relationship(self.unit1)
        create_many_to_one_relationship(self.unit2)

    def setup_analogsignals(self):
        signame11 = 'analogsignal 1 1'
        signame12 = 'analogsignal 1 2'
        signame21 = 'analogsignal 2 1'
        signame22 = 'analogsignal 2 2'

        sigdata11 = np.arange(0, 10) * pq.mV
        sigdata12 = np.arange(10, 20) * pq.mV
        sigdata21 = np.arange(20, 30) * pq.V
        sigdata22 = np.arange(30, 40) * pq.V

        self.signames1 = [signame11, signame12]
        self.signames2 = [signame21, signame22]
        self.signames = [signame11, signame12, signame21, signame22]

        sig11 = AnalogSignal(sigdata11, name=signame11,
                             channel_index=1, sampling_rate=1*pq.Hz)
        sig12 = AnalogSignal(sigdata12, name=signame12,
                             channel_index=2, sampling_rate=1*pq.Hz)
        sig21 = AnalogSignal(sigdata21, name=signame21,
                             channel_index=1, sampling_rate=1*pq.Hz)
        sig22 = AnalogSignal(sigdata22, name=signame22,
                             channel_index=2, sampling_rate=1*pq.Hz)

        self.sig1 = [sig11, sig12]
        self.sig2 = [sig21, sig22]
        self.sig = [sig11, sig12, sig21, sig22]

        self.chan1sig = [self.sig1[0], self.sig2[0]]
        self.chan2sig = [self.sig1[1], self.sig2[1]]

    def setup_analogsignalarrays(self):
        sigarrname11 = 'analogsignalarray 1 1'
        sigarrname12 = 'analogsignalarray 1 2'
        sigarrname21 = 'analogsignalarray 2 1'
        sigarrname22 = 'analogsignalarray 2 2'

        sigarrdata11 = np.arange(0, 10).reshape(5, 2) * pq.mV
        sigarrdata12 = np.arange(10, 20).reshape(5, 2) * pq.mV
        sigarrdata21 = np.arange(20, 30).reshape(5, 2) * pq.V
        sigarrdata22 = np.arange(30, 40).reshape(5, 2) * pq.V
        sigarrdata112 = np.hstack([sigarrdata11, sigarrdata11]) * pq.mV

        self.sigarrnames1 = [sigarrname11, sigarrname12]
        self.sigarrnames2 = [sigarrname21, sigarrname22, sigarrname11]
        self.sigarrnames = [sigarrname11, sigarrname12,
                            sigarrname21, sigarrname22]

        sigarr11 = AnalogSignalArray(sigarrdata11, name=sigarrname11,
                                     sampling_rate=1*pq.Hz,
                                     channel_index=np.array([1, 2]))
        sigarr12 = AnalogSignalArray(sigarrdata12, name=sigarrname12,
                                     sampling_rate=1*pq.Hz,
                                     channel_index=np.array([2, 1]))
        sigarr21 = AnalogSignalArray(sigarrdata21, name=sigarrname21,
                                     sampling_rate=1*pq.Hz,
                                     channel_index=np.array([1, 2]))
        sigarr22 = AnalogSignalArray(sigarrdata22, name=sigarrname22,
                                     sampling_rate=1*pq.Hz,
                                     channel_index=np.array([2, 1]))
        sigarr23 = AnalogSignalArray(sigarrdata11, name=sigarrname11,
                                     sampling_rate=1*pq.Hz,
                                     channel_index=np.array([1, 2]))
        sigarr112 = AnalogSignalArray(sigarrdata112, name=sigarrname11,
                                      sampling_rate=1*pq.Hz,
                                      channel_index=np.array([1, 2]))

        self.sigarr1 = [sigarr11, sigarr12]
        self.sigarr2 = [sigarr21, sigarr22, sigarr23]
        self.sigarr = [sigarr112, sigarr12, sigarr21, sigarr22]

        self.chan1sigarr1 = [sigarr11[:, 0:1], sigarr12[:, 1:2]]
        self.chan2sigarr1 = [sigarr11[:, 1:2], sigarr12[:, 0:1]]
        self.chan1sigarr2 = [sigarr21[:, 0:1], sigarr22[:, 1:2],
                             sigarr23[:, 0:1]]
        self.chan2sigarr2 = [sigarr21[:, 1:2], sigarr22[:, 0:1],
                             sigarr23[:, 0:1]]

    def setup_epochs(self):
        epochname11 = 'epoch 1 1'
        epochname12 = 'epoch 1 2'
        epochname21 = 'epoch 2 1'
        epochname22 = 'epoch 2 2'

        epochtime11 = 10 * pq.ms
        epochtime12 = 20 * pq.ms
        epochtime21 = 30 * pq.s
        epochtime22 = 40 * pq.s

        epochdur11 = 11 * pq.s
        epochdur12 = 21 * pq.s
        epochdur21 = 31 * pq.ms
        epochdur22 = 41 * pq.ms

        self.epochnames1 = [epochname11, epochname12]
        self.epochnames2 = [epochname21, epochname22]
        self.epochnames = [epochname11, epochname12, epochname21, epochname22]

        epoch11 = Epoch(epochtime11, epochdur11,
                        label=epochname11, name=epochname11, channel_index=1,
                        testattr=True)
        epoch12 = Epoch(epochtime12, epochdur12,
                        label=epochname12, name=epochname12, channel_index=2,
                        testattr=False)
        epoch21 = Epoch(epochtime21, epochdur21,
                        label=epochname21, name=epochname21, channel_index=1)
        epoch22 = Epoch(epochtime22, epochdur22,
                        label=epochname22, name=epochname22, channel_index=2)

        self.epoch1 = [epoch11, epoch12]
        self.epoch2 = [epoch21, epoch22]
        self.epoch = [epoch11, epoch12, epoch21, epoch22]

    def setup_epocharrays(self):
        epocharrname11 = 'epocharr 1 1'
        epocharrname12 = 'epocharr 1 2'
        epocharrname21 = 'epocharr 2 1'
        epocharrname22 = 'epocharr 2 2'

        epocharrtime11 = np.arange(0, 10) * pq.ms
        epocharrtime12 = np.arange(10, 20) * pq.ms
        epocharrtime21 = np.arange(20, 30) * pq.s
        epocharrtime22 = np.arange(30, 40) * pq.s

        epocharrdur11 = np.arange(1, 11) * pq.s
        epocharrdur12 = np.arange(11, 21) * pq.s
        epocharrdur21 = np.arange(21, 31) * pq.ms
        epocharrdur22 = np.arange(31, 41) * pq.ms

        self.epocharrnames1 = [epocharrname11, epocharrname12]
        self.epocharrnames2 = [epocharrname21, epocharrname22]
        self.epocharrnames = [epocharrname11,
                              epocharrname12, epocharrname21, epocharrname22]

        epocharr11 = EpochArray(epocharrtime11, epocharrdur11,
                                label=epocharrname11, name=epocharrname11)
        epocharr12 = EpochArray(epocharrtime12, epocharrdur12,
                                label=epocharrname12, name=epocharrname12)
        epocharr21 = EpochArray(epocharrtime21, epocharrdur21,
                                label=epocharrname21, name=epocharrname21)
        epocharr22 = EpochArray(epocharrtime22, epocharrdur22,
                                label=epocharrname22, name=epocharrname22)

        self.epocharr1 = [epocharr11, epocharr12]
        self.epocharr2 = [epocharr21, epocharr22]
        self.epocharr = [epocharr11, epocharr12, epocharr21, epocharr22]

    def setup_events(self):
        eventname11 = 'event 1 1'
        eventname12 = 'event 1 2'
        eventname21 = 'event 2 1'
        eventname22 = 'event 2 2'

        eventtime11 = 10 * pq.ms
        eventtime12 = 20 * pq.ms
        eventtime21 = 30 * pq.s
        eventtime22 = 40 * pq.s

        self.eventnames1 = [eventname11, eventname12]
        self.eventnames2 = [eventname21, eventname22]
        self.eventnames = [eventname11, eventname12, eventname21, eventname22]

        params1 = {'testattr': True}
        params2 = {'testattr': 5}
        event11 = Event(eventtime11, label=eventname11, name=eventname11,
                        **params1)
        event12 = Event(eventtime12, label=eventname12, name=eventname12,
                        **params2)
        event21 = Event(eventtime21, label=eventname21, name=eventname21)
        event22 = Event(eventtime22, label=eventname22, name=eventname22)

        self.event1 = [event11, event12]
        self.event2 = [event21, event22]
        self.event = [event11, event12, event21, event22]

    def setup_eventarrays(self):
        eventarrname11 = 'eventarr 1 1'
        eventarrname12 = 'eventarr 1 2'
        eventarrname21 = 'eventarr 2 1'
        eventarrname22 = 'eventarr 2 2'

        eventarrtime11 = np.arange(0, 10) * pq.ms
        eventarrtime12 = np.arange(10, 20) * pq.ms
        eventarrtime21 = np.arange(20, 30) * pq.s
        eventarrtime22 = np.arange(30, 40) * pq.s

        self.eventarrnames1 = [eventarrname11, eventarrname12]
        self.eventarrnames2 = [eventarrname21, eventarrname22]
        self.eventarrnames = [eventarrname11,
                              eventarrname12, eventarrname21, eventarrname22]

        eventarr11 = EventArray(eventarrtime11,
                                label=eventarrname11, name=eventarrname11)
        eventarr12 = EventArray(eventarrtime12,
                                label=eventarrname12, name=eventarrname12)
        eventarr21 = EventArray(eventarrtime21,
                                label=eventarrname21, name=eventarrname21)
        eventarr22 = EventArray(eventarrtime22,
                                label=eventarrname22, name=eventarrname22)

        self.eventarr1 = [eventarr11, eventarr12]
        self.eventarr2 = [eventarr21, eventarr22]
        self.eventarr = [eventarr11, eventarr12, eventarr21, eventarr22]

    def setup_irregularlysampledsignals(self):
        irsigname11 = 'irregularsignal 1 1'
        irsigname12 = 'irregularsignal 1 2'
        irsigname21 = 'irregularsignal 2 1'
        irsigname22 = 'irregularsignal 2 2'

        irsigdata11 = np.arange(0, 10) * pq.mA
        irsigdata12 = np.arange(10, 20) * pq.mA
        irsigdata21 = np.arange(20, 30) * pq.A
        irsigdata22 = np.arange(30, 40) * pq.A

        irsigtimes11 = np.arange(0, 10) * pq.ms
        irsigtimes12 = np.arange(10, 20) * pq.ms
        irsigtimes21 = np.arange(20, 30) * pq.s
        irsigtimes22 = np.arange(30, 40) * pq.s

        self.irsignames1 = [irsigname11, irsigname12]
        self.irsignames2 = [irsigname21, irsigname22]
        self.irsignames = [irsigname11, irsigname12, irsigname21, irsigname22]

        irsig11 = IrregularlySampledSignal(irsigtimes11, irsigdata11,
                                           name=irsigname11)
        irsig12 = IrregularlySampledSignal(irsigtimes12, irsigdata12,
                                           name=irsigname12)
        irsig21 = IrregularlySampledSignal(irsigtimes21, irsigdata21,
                                           name=irsigname21)
        irsig22 = IrregularlySampledSignal(irsigtimes22, irsigdata22,
                                           name=irsigname22)

        self.irsig1 = [irsig11, irsig12]
        self.irsig2 = [irsig21, irsig22]
        self.irsig = [irsig11, irsig12, irsig21, irsig22]

    def setup_spikes(self):
        spikename11 = 'spike 1 1'
        spikename12 = 'spike 1 2'
        spikename21 = 'spike 2 1'
        spikename22 = 'spike 2 2'

        spikedata11 = 10 * pq.ms
        spikedata12 = 20 * pq.ms
        spikedata21 = 30 * pq.s
        spikedata22 = 40 * pq.s

        self.spikenames1 = [spikename11, spikename12]
        self.spikenames2 = [spikename21, spikename22]
        self.spikenames = [spikename11, spikename12, spikename21, spikename22]

        spike11 = Spike(spikedata11, t_stop=100*pq.s, name=spikename11)
        spike12 = Spike(spikedata12, t_stop=100*pq.s, name=spikename12)
        spike21 = Spike(spikedata21, t_stop=100*pq.s, name=spikename21)
        spike22 = Spike(spikedata22, t_stop=100*pq.s, name=spikename22)

        self.spike1 = [spike11, spike12]
        self.spike2 = [spike21, spike22]
        self.spike = [spike11, spike12, spike21, spike22]

    def setup_spiketrains(self):
        trainname11 = 'spiketrain 1 1'
        trainname12 = 'spiketrain 1 2'
        trainname21 = 'spiketrain 2 1'
        trainname22 = 'spiketrain 2 2'

        traindata11 = np.arange(0, 10) * pq.ms
        traindata12 = np.arange(10, 20) * pq.ms
        traindata21 = np.arange(20, 30) * pq.s
        traindata22 = np.arange(30, 40) * pq.s

        self.trainnames1 = [trainname11, trainname12]
        self.trainnames2 = [trainname21, trainname22]
        self.trainnames = [trainname11, trainname12, trainname21, trainname22]

        train11 = SpikeTrain(traindata11, t_stop=100*pq.s, name=trainname11)
        train12 = SpikeTrain(traindata12, t_stop=100*pq.s, name=trainname12)
        train21 = SpikeTrain(traindata21, t_stop=100*pq.s, name=trainname21)
        train22 = SpikeTrain(traindata22, t_stop=100*pq.s, name=trainname22)

        self.train1 = [train11, train12]
        self.train2 = [train21, train22]
        self.train = [train11, train12, train21, train22]

    def test_init(self):
        seg = Segment(name='a segment', index=3)
        assert_neo_object_is_compliant(seg)
        self.assertEqual(seg.name, 'a segment')
        self.assertEqual(seg.file_origin, None)
        self.assertEqual(seg.index, 3)

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
                st = SpikeTrain([1, 2, 3], units='ms',
                                t_start=0., t_stop=10)
                st.unit = all_unit[j]

            for t in signal_types:
                anasigarr = AnalogSignalArray(np.zeros((sig_len,
                                                        len(unit_with_sig))),
                                              units='nA',
                                              sampling_rate=1000.*pq.Hz,
                                              channel_indexes=unit_with_sig)
                seg.analogsignalarrays.append(anasigarr)

        create_many_to_one_relationship(blk)
        for unit in all_unit:
            assert_neo_object_is_compliant(unit)
        for rcg in rcgs:
            assert_neo_object_is_compliant(rcg)
        assert_neo_object_is_compliant(blk)

        # what you want
        newseg = seg.construct_subsegment_by_unit(all_unit[:4])
        assert_neo_object_is_compliant(newseg)

    def test_segment_creation(self):
        assert_neo_object_is_compliant(self.segment1)
        assert_neo_object_is_compliant(self.segment2)
        assert_neo_object_is_compliant(self.unit1)
        assert_neo_object_is_compliant(self.unit2)

        self.assertEqual(self.segment1.name, 'test')
        self.assertEqual(self.segment2.name, 'test')

        self.assertEqual(self.segment1.description, 'tester 1')
        self.assertEqual(self.segment2.description, 'tester 2')

        self.assertEqual(self.segment1.file_origin, 'test.file')
        self.assertEqual(self.segment2.file_origin, 'test.file')

        self.assertEqual(self.segment1.annotations['testarg0'], [1, 2, 3])
        self.assertEqual(self.segment2.annotations['testarg10'], [1, 2, 3])

        self.assertEqual(self.segment1.annotations['testarg1'], 1.1)
        self.assertEqual(self.segment2.annotations['testarg1'], 1)
        self.assertEqual(self.segment2.annotations['testarg11'], 1.1)

        self.assertEqual(self.segment1.annotations['testarg2'], 'yes')
        self.assertEqual(self.segment2.annotations['testarg2'], 'yes')

        self.assertTrue(self.segment1.annotations['testarg3'])
        self.assertTrue(self.segment2.annotations['testarg3'])

        self.assertTrue(hasattr(self.segment1, 'analogsignals'))
        self.assertTrue(hasattr(self.segment2, 'analogsignals'))

        self.assertEqual(len(self.segment1.analogsignals), 2)
        self.assertEqual(len(self.segment2.analogsignals), 2)

        for res, targ in zip(self.segment1.analogsignals, self.sig1):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.analogsignals, self.sig2):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.segment1, 'analogsignalarrays'))
        self.assertTrue(hasattr(self.segment2, 'analogsignalarrays'))

        self.assertEqual(len(self.segment1.analogsignalarrays), 2)
        self.assertEqual(len(self.segment2.analogsignalarrays), 3)

        for res, targ in zip(self.segment1.analogsignalarrays, self.sigarr1):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.analogsignalarrays, self.sigarr2):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.segment1, 'epochs'))
        self.assertTrue(hasattr(self.segment2, 'epochs'))

        self.assertEqual(len(self.segment1.epochs), 2)
        self.assertEqual(len(self.segment2.epochs), 2)

        for res, targ in zip(self.segment1.epochs, self.epoch1):
            self.assertEqual(res.time, targ.time)
            self.assertEqual(res.duration, targ.duration)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.epochs, self.epoch2):
            self.assertEqual(res.time, targ.time)
            self.assertEqual(res.duration, targ.duration)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.segment1, 'epocharrays'))
        self.assertTrue(hasattr(self.segment2, 'epocharrays'))

        self.assertEqual(len(self.segment1.epocharrays), 2)
        self.assertEqual(len(self.segment2.epocharrays), 2)

        for res, targ in zip(self.segment1.epocharrays, self.epocharr1):
            assert_arrays_equal(res.times, targ.times)
            assert_arrays_equal(res.durations, targ.durations)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.epocharrays, self.epocharr2):
            assert_arrays_equal(res.times, targ.times)
            assert_arrays_equal(res.durations, targ.durations)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.segment1, 'events'))
        self.assertTrue(hasattr(self.segment2, 'events'))

        self.assertEqual(len(self.segment1.events), 2)
        self.assertEqual(len(self.segment2.events), 2)

        for res, targ in zip(self.segment1.events, self.event1):
            self.assertEqual(res.time, targ.time)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.events, self.event2):
            self.assertEqual(res.time, targ.time)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.segment1, 'eventarrays'))
        self.assertTrue(hasattr(self.segment2, 'eventarrays'))

        self.assertEqual(len(self.segment1.eventarrays), 2)
        self.assertEqual(len(self.segment2.eventarrays), 2)

        for res, targ in zip(self.segment1.eventarrays, self.eventarr1):
            assert_arrays_equal(res.times, targ.times)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.eventarrays, self.eventarr2):
            assert_arrays_equal(res.times, targ.times)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.segment1, 'irregularlysampledsignals'))
        self.assertTrue(hasattr(self.segment2, 'irregularlysampledsignals'))

        self.assertEqual(len(self.segment1.irregularlysampledsignals), 2)
        self.assertEqual(len(self.segment2.irregularlysampledsignals), 2)

        for res, targ in zip(self.segment1.irregularlysampledsignals,
                             self.irsig1):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.irregularlysampledsignals,
                             self.irsig2):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.segment1, 'spikes'))
        self.assertTrue(hasattr(self.segment2, 'spikes'))

        self.assertEqual(len(self.segment1.spikes), 2)
        self.assertEqual(len(self.segment2.spikes), 2)

        for res, targ in zip(self.segment1.spikes, self.spike1):
            self.assertEqual(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.spikes, self.spike2):
            self.assertEqual(res, targ)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.segment1, 'spiketrains'))
        self.assertTrue(hasattr(self.segment2, 'spiketrains'))

        self.assertEqual(len(self.segment1.spiketrains), 2)
        self.assertEqual(len(self.segment2.spiketrains), 2)

        for res, targ in zip(self.segment1.spiketrains, self.train1):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.spiketrains, self.train2):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

    def test_segment_merge(self):
        self.segment1.merge(self.segment2)
        create_many_to_one_relationship(self.segment1, force=True)
        assert_neo_object_is_compliant(self.segment1)

        self.assertEqual(self.segment1.name, 'test')
        self.assertEqual(self.segment2.name, 'test')

        self.assertEqual(self.segment1.description, 'tester 1')
        self.assertEqual(self.segment2.description, 'tester 2')

        self.assertEqual(self.segment1.file_origin, 'test.file')
        self.assertEqual(self.segment2.file_origin, 'test.file')

        self.assertEqual(self.segment1.annotations['testarg0'], [1, 2, 3])
        self.assertEqual(self.segment2.annotations['testarg10'], [1, 2, 3])

        self.assertEqual(self.segment1.annotations['testarg1'], 1.1)
        self.assertEqual(self.segment2.annotations['testarg1'], 1)
        self.assertEqual(self.segment2.annotations['testarg11'], 1.1)

        self.assertEqual(self.segment1.annotations['testarg2'], 'yes')
        self.assertEqual(self.segment2.annotations['testarg2'], 'yes')

        self.assertTrue(self.segment1.annotations['testarg3'])
        self.assertTrue(self.segment2.annotations['testarg3'])

        self.assertTrue(hasattr(self.segment1, 'analogsignals'))
        self.assertTrue(hasattr(self.segment2, 'analogsignals'))

        self.assertEqual(len(self.segment1.analogsignals), 4)
        self.assertEqual(len(self.segment2.analogsignals), 2)

        for res, targ in zip(self.segment1.analogsignals, self.sig):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.analogsignals, self.sig2):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.segment1, 'analogsignalarrays'))
        self.assertTrue(hasattr(self.segment2, 'analogsignalarrays'))

        self.assertEqual(len(self.segment1.analogsignalarrays), 4)
        self.assertEqual(len(self.segment2.analogsignalarrays), 3)

        for res, targ in zip(self.segment1.analogsignalarrays, self.sigarr):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.analogsignalarrays, self.sigarr2):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.segment1, 'epochs'))
        self.assertTrue(hasattr(self.segment2, 'epochs'))

        self.assertEqual(len(self.segment1.epochs), 4)
        self.assertEqual(len(self.segment2.epochs), 2)

        for res, targ in zip(self.segment1.epochs, self.epoch):
            self.assertEqual(res.time, targ.time)
            self.assertEqual(res.duration, targ.duration)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.epochs, self.epoch2):
            self.assertEqual(res.time, targ.time)
            self.assertEqual(res.duration, targ.duration)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.segment1, 'epocharrays'))
        self.assertTrue(hasattr(self.segment2, 'epocharrays'))

        self.assertEqual(len(self.segment1.epocharrays), 4)
        self.assertEqual(len(self.segment2.epocharrays), 2)

        for res, targ in zip(self.segment1.epocharrays, self.epocharr):
            assert_arrays_equal(res.times, targ.times)
            assert_arrays_equal(res.durations, targ.durations)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.epocharrays, self.epocharr2):
            assert_arrays_equal(res.times, targ.times)
            assert_arrays_equal(res.durations, targ.durations)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.segment1, 'events'))
        self.assertTrue(hasattr(self.segment2, 'events'))

        self.assertEqual(len(self.segment1.events), 4)
        self.assertEqual(len(self.segment2.events), 2)

        for res, targ in zip(self.segment1.events, self.event):
            self.assertEqual(res.time, targ.time)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.events, self.event2):
            self.assertEqual(res.time, targ.time)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.segment1, 'eventarrays'))
        self.assertTrue(hasattr(self.segment2, 'eventarrays'))

        self.assertEqual(len(self.segment1.eventarrays), 4)
        self.assertEqual(len(self.segment2.eventarrays), 2)

        for res, targ in zip(self.segment1.eventarrays, self.eventarr):
            assert_arrays_equal(res.times, targ.times)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.eventarrays, self.eventarr2):
            assert_arrays_equal(res.times, targ.times)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.segment1, 'irregularlysampledsignals'))
        self.assertTrue(hasattr(self.segment2, 'irregularlysampledsignals'))

        self.assertEqual(len(self.segment1.irregularlysampledsignals), 4)
        self.assertEqual(len(self.segment2.irregularlysampledsignals), 2)

        for res, targ in zip(self.segment1.irregularlysampledsignals,
                             self.irsig):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.irregularlysampledsignals,
                             self.irsig2):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.segment1, 'spikes'))
        self.assertTrue(hasattr(self.segment2, 'spikes'))

        self.assertEqual(len(self.segment1.spikes), 4)
        self.assertEqual(len(self.segment2.spikes), 2)

        for res, targ in zip(self.segment1.spikes, self.spike):
            self.assertEqual(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.spikes, self.spike2):
            self.assertEqual(res, targ)
            self.assertEqual(res.name, targ.name)

        self.assertTrue(hasattr(self.segment1, 'spiketrains'))
        self.assertTrue(hasattr(self.segment2, 'spiketrains'))

        self.assertEqual(len(self.segment1.spiketrains), 4)
        self.assertEqual(len(self.segment2.spiketrains), 2)

        for res, targ in zip(self.segment1.spiketrains, self.train):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(self.segment2.spiketrains, self.train2):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

    def test_segment_all_data(self):
        result1 = self.segment1.all_data
        targs = (self.epoch1 + self.epocharr1 + self.event1 + self.eventarr1 +
                 self.sig1 + self.sigarr1 + self.irsig1 +
                 self.spike1 + self.train1)

        for res, targ in zip(result1, targs):
            if hasattr(res, 'ndim') and res.ndim:
                assert_arrays_equal(res, targ)
            else:
                self.assertEqual(res, targ)
            self.assertEqual(res.name, targ.name)

    def test_segment_take_spikes_by_unit(self):
        result1 = self.segment1.take_spikes_by_unit()
        result21 = self.segment1.take_spikes_by_unit([self.unit1])
        result22 = self.segment1.take_spikes_by_unit([self.unit2])

        self.assertEqual(result1, [])

        for res, targ in zip(result21, self.unit1spike):
            self.assertEqual(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(result22, self.unit2spike):
            self.assertEqual(res, targ)
            self.assertEqual(res.name, targ.name)

    def test_segment_take_spiketrains_by_unit(self):
        result1 = self.segment1.take_spiketrains_by_unit()
        result21 = self.segment1.take_spiketrains_by_unit([self.unit1])
        result22 = self.segment1.take_spiketrains_by_unit([self.unit2])

        self.assertEqual(result1, [])

        for res, targ in zip(result21, self.unit1train):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(result22, self.unit2train):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

    def test_segment_take_analogsignal_by_unit(self):
        result1 = self.segment1.take_analogsignal_by_unit()
        result21 = self.segment1.take_analogsignal_by_unit([self.unit1])
        result22 = self.segment1.take_analogsignal_by_unit([self.unit2])

        self.assertEqual(result1, [])

        for res, targ in zip(result21, self.chan1sig):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(result22, self.chan2sig):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

    def test_segment_take_analogsignal_by_channelindex(self):
        result1 = self.segment1.take_analogsignal_by_channelindex()
        result21 = self.segment1.take_analogsignal_by_channelindex([1])
        result22 = self.segment1.take_analogsignal_by_channelindex([2])

        self.assertEqual(result1, [])

        for res, targ in zip(result21, self.chan1sig):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(result22, self.chan2sig):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

    def test_segment_take_slice_of_analogsignalarray_by_unit(self):
        segment = self.segment1
        unit1 = self.unit1
        unit2 = self.unit2

        result1 = segment.take_slice_of_analogsignalarray_by_unit()
        result21 = segment.take_slice_of_analogsignalarray_by_unit([unit1])
        result22 = segment.take_slice_of_analogsignalarray_by_unit([unit2])

        self.assertEqual(result1, [])

        for res, targ in zip(result21, self.chan1sigarr1):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(result22, self.chan2sigarr1):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

    def test_segment_take_slice_of_analogsignalarray_by_channelindex(self):
        segment = self.segment1
        result1 = segment.take_slice_of_analogsignalarray_by_channelindex()
        result21 = segment.take_slice_of_analogsignalarray_by_channelindex([1])
        result22 = segment.take_slice_of_analogsignalarray_by_channelindex([2])

        self.assertEqual(result1, [])

        for res, targ in zip(result21, self.chan1sigarr1):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

        for res, targ in zip(result22, self.chan2sigarr1):
            assert_arrays_equal(res, targ)
            self.assertEqual(res.name, targ.name)

    def test_segment_size(self):
        result1 = self.segment1.size()
        targ1 = {"epochs": 2,  "events": 2,  "analogsignals": 2,
                 "irregularlysampledsignals": 2, "spikes": 2,
                 "spiketrains": 2, "epocharrays": 2, "eventarrays": 2,
                 "analogsignalarrays": 2}

        self.assertEqual(result1, targ1)

    def test_segment_filter(self):
        result1 = self.segment1.filter()
        result2 = self.segment1.filter(name='analogsignal 1 1')
        result3 = self.segment1.filter(testattr=True)

        self.assertEqual(result1, [])

        self.assertEqual(len(result2), 1)
        assert_arrays_equal(result2[0], self.sig1[0])

        self.assertEqual(len(result3), 2)
        self.assertEqual(result3[0], self.epoch1[0])
        self.assertEqual(result3[1], self.event1[0])


if __name__ == "__main__":
    unittest.main()
