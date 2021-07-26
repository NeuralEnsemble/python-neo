"""
Tests of the neo.utils.misc module
"""

import unittest
import warnings

import numpy as np
import quantities as pq
from neo.rawio.examplerawio import ExampleRawIO
from neo.io.proxyobjects import (AnalogSignalProxy, SpikeTrainProxy,
                EventProxy, EpochProxy)

from neo.core.dataobject import ArrayDict
from neo.core import (Block, Segment, AnalogSignal, IrregularlySampledSignal,
                      Epoch, Event, SpikeTrain)

from neo.test.tools import (assert_arrays_almost_equal,
                            assert_arrays_equal,
                            assert_neo_object_is_compliant,
                            assert_same_attributes,
                            assert_same_annotations)

from neo.utils import (get_events, get_epochs, add_epoch, match_events, cut_block_by_epochs)


class BaseProxyTest(unittest.TestCase):
    def setUp(self):
        self.reader = ExampleRawIO(filename='my_filename.fake')
        self.reader.parse_header()


class TestUtilsWithoutProxyObjects(unittest.TestCase):
    def test__get_events(self):
        starts_1 = Event(times=[0.5, 10.0, 25.2] * pq.s,
                         labels=['label1', 'label2', 'label3'],
                         name='pick_me')
        starts_1.annotate(event_type='trial start')
        starts_1.array_annotate(trial_id=[1, 2, 3])

        stops_1 = Event(times=[5.5, 14.9, 30.1] * pq.s)
        stops_1.annotate(event_type='trial stop')
        stops_1.array_annotate(trial_id=[1, 2, 3])

        starts_2 = Event(times=[33.2, 41.7, 52.4] * pq.s)
        starts_2.annotate(event_type='trial start')
        starts_2.array_annotate(trial_id=[4, 5, 6])

        stops_2 = Event(times=[37.6, 46.1, 57.0] * pq.s)
        stops_2.annotate(event_type='trial stop')
        stops_2.array_annotate(trial_id=[4, 5, 6])

        seg = Segment()
        seg2 = Segment()
        seg.events = [starts_1, stops_1]
        seg2.events = [starts_2, stops_2]

        block = Block()
        block.segments = [seg, seg2]

        # test getting one whole event via annotation or attribute
        extracted_starts1 = get_events(seg, event_type='trial start')
        extracted_starts1b = get_events(block, name='pick_me')

        self.assertEqual(len(extracted_starts1), 1)
        self.assertEqual(len(extracted_starts1b), 1)

        extracted_starts1 = extracted_starts1[0]
        extracted_starts1b = extracted_starts1b[0]

        assert_same_attributes(extracted_starts1, starts_1)
        assert_same_attributes(extracted_starts1b, starts_1)

        # test getting an empty list by searching for a non-existent property
        empty1 = get_events(seg, foo='bar')

        self.assertEqual(len(empty1), 0)

        # test getting an empty list by searching for a non-existent property value
        empty2 = get_events(seg, event_type='undefined')

        self.assertEqual(len(empty2), 0)

        # test getting only one event time of one event
        trial_2 = get_events(block, trial_id=2, event_type='trial start')

        self.assertEqual(len(trial_2), 1)

        trial_2 = trial_2[0]

        self.assertEqual(starts_1.name, trial_2.name)
        self.assertEqual(starts_1.description, trial_2.description)
        self.assertEqual(starts_1.file_origin, trial_2.file_origin)
        self.assertEqual(starts_1.annotations['event_type'], trial_2.annotations['event_type'])
        assert_arrays_equal(trial_2.array_annotations['trial_id'], np.array([2]))
        self.assertIsInstance(trial_2.array_annotations, ArrayDict)

        # test getting only one event time of more than one event
        trial_2b = get_events(block, trial_id=2)

        self.assertEqual(len(trial_2b), 2)

        start_idx = np.where(np.array([ev.annotations['event_type']
                                       for ev in trial_2b]) == 'trial start')[0][0]

        trial_2b_start = trial_2b[start_idx]
        trial_2b_stop = trial_2b[start_idx - 1]

        assert_same_attributes(trial_2b_start, trial_2)

        self.assertEqual(stops_1.name, trial_2b_stop.name)
        self.assertEqual(stops_1.description, trial_2b_stop.description)
        self.assertEqual(stops_1.file_origin, trial_2b_stop.file_origin)
        self.assertEqual(stops_1.annotations['event_type'],
                         trial_2b_stop.annotations['event_type'])
        assert_arrays_equal(trial_2b_stop.array_annotations['trial_id'], np.array([2]))
        self.assertIsInstance(trial_2b_stop.array_annotations, ArrayDict)

        # test getting more than one event time of one event
        trials_1_2 = get_events(block, trial_id=[1, 2], event_type='trial start')

        self.assertEqual(len(trials_1_2), 1)

        trials_1_2 = trials_1_2[0]

        self.assertEqual(starts_1.name, trials_1_2.name)
        self.assertEqual(starts_1.description, trials_1_2.description)
        self.assertEqual(starts_1.file_origin, trials_1_2.file_origin)
        self.assertEqual(starts_1.annotations['event_type'], trials_1_2.annotations['event_type'])
        assert_arrays_equal(trials_1_2.array_annotations['trial_id'], np.array([1, 2]))
        self.assertIsInstance(trials_1_2.array_annotations, ArrayDict)

        # test selecting event times by label
        trials_1_2 = get_events(block, labels=['label1', 'label2'])

        self.assertEqual(len(trials_1_2), 1)

        trials_1_2 = trials_1_2[0]

        self.assertEqual(starts_1.name, trials_1_2.name)
        self.assertEqual(starts_1.description, trials_1_2.description)
        self.assertEqual(starts_1.file_origin, trials_1_2.file_origin)
        self.assertEqual(starts_1.annotations['event_type'], trials_1_2.annotations['event_type'])
        assert_arrays_equal(trials_1_2.array_annotations['trial_id'], np.array([1, 2]))
        self.assertIsInstance(trials_1_2.array_annotations, ArrayDict)

        # test getting more than one event time of more than one event
        trials_1_2b = get_events(block, trial_id=[1, 2])

        self.assertEqual(len(trials_1_2b), 2)

        start_idx = np.where(np.array([ev.annotations['event_type']
                                       for ev in trials_1_2b]) == 'trial start')[0][0]

        trials_1_2b_start = trials_1_2b[start_idx]
        trials_1_2b_stop = trials_1_2b[start_idx - 1]

        assert_same_attributes(trials_1_2b_start, trials_1_2)

        self.assertEqual(stops_1.name, trials_1_2b_stop.name)
        self.assertEqual(stops_1.description, trials_1_2b_stop.description)
        self.assertEqual(stops_1.file_origin, trials_1_2b_stop.file_origin)
        self.assertEqual(stops_1.annotations['event_type'],
                         trials_1_2b_stop.annotations['event_type'])
        assert_arrays_equal(trials_1_2b_stop.array_annotations['trial_id'], np.array([1, 2]))
        self.assertIsInstance(trials_1_2b_stop.array_annotations, ArrayDict)

    def test__get_epochs(self):
        a_1 = Epoch([0.5, 10.0, 25.2] * pq.s, durations=[5.1, 4.8, 5.0] * pq.s)
        a_1.annotate(epoch_type='a', pick='me')
        a_1.array_annotate(trial_id=[1, 2, 3])

        b_1 = Epoch([5.5, 14.9, 30.1] * pq.s, durations=[4.7, 4.9, 5.2] * pq.s)
        b_1.annotate(epoch_type='b')
        b_1.array_annotate(trial_id=[1, 2, 3])

        a_2 = Epoch([33.2, 41.7, 52.4] * pq.s, durations=[5.3, 5.0, 5.1] * pq.s)
        a_2.annotate(epoch_type='a')
        a_2.array_annotate(trial_id=[4, 5, 6])

        b_2 = Epoch([37.6, 46.1, 57.0] * pq.s, durations=[4.9, 5.2, 5.1] * pq.s)
        b_2.annotate(epoch_type='b')
        b_2.array_annotate(trial_id=[4, 5, 6])

        seg = Segment()
        seg2 = Segment()
        seg.epochs = [a_1, b_1]
        seg2.epochs = [a_2, b_2]

        block = Block()
        block.segments = [seg, seg2]

        # test getting one whole event via annotation
        extracted_a_1 = get_epochs(seg, epoch_type='a')
        extracted_a_1b = get_epochs(block, pick='me')

        self.assertEqual(len(extracted_a_1), 1)
        self.assertEqual(len(extracted_a_1b), 1)

        extracted_a_1 = extracted_a_1[0]
        extracted_a_1b = extracted_a_1b[0]

        assert_same_attributes(extracted_a_1, a_1)
        assert_same_attributes(extracted_a_1b, a_1)

        # test getting an empty list by searching for a non-existent property
        empty1 = get_epochs(seg, foo='bar')

        self.assertEqual(len(empty1), 0)

        # test getting an empty list by searching for a non-existent property value
        empty2 = get_epochs(seg, epoch_type='undefined')

        self.assertEqual(len(empty2), 0)

        # test getting only one event time of one event
        trial_2 = get_epochs(block, trial_id=2, epoch_type='a')

        self.assertEqual(len(trial_2), 1)

        trial_2 = trial_2[0]

        self.assertEqual(a_1.name, trial_2.name)
        self.assertEqual(a_1.description, trial_2.description)
        self.assertEqual(a_1.file_origin, trial_2.file_origin)
        self.assertEqual(a_1.annotations['epoch_type'], trial_2.annotations['epoch_type'])
        assert_arrays_equal(trial_2.array_annotations['trial_id'], np.array([2]))
        self.assertIsInstance(trial_2.array_annotations, ArrayDict)

        # test getting only one event time of more than one event
        trial_2b = get_epochs(block, trial_id=2)

        self.assertEqual(len(trial_2b), 2)

        a_idx = np.where(np.array([ev.annotations['epoch_type'] for ev in trial_2b]) == 'a')[0][0]

        trial_2b_a = trial_2b[a_idx]
        trial_2b_b = trial_2b[a_idx - 1]

        assert_same_attributes(trial_2b_a, trial_2)

        self.assertEqual(b_1.name, trial_2b_b.name)
        self.assertEqual(b_1.description, trial_2b_b.description)
        self.assertEqual(b_1.file_origin, trial_2b_b.file_origin)
        self.assertEqual(b_1.annotations['epoch_type'], trial_2b_b.annotations['epoch_type'])
        assert_arrays_equal(trial_2b_b.array_annotations['trial_id'], np.array([2]))
        self.assertIsInstance(trial_2b_b.array_annotations, ArrayDict)

        # test getting more than one event time of one event
        trials_1_2 = get_epochs(block, trial_id=[1, 2], epoch_type='a')

        self.assertEqual(len(trials_1_2), 1)

        trials_1_2 = trials_1_2[0]

        self.assertEqual(a_1.name, trials_1_2.name)
        self.assertEqual(a_1.description, trials_1_2.description)
        self.assertEqual(a_1.file_origin, trials_1_2.file_origin)
        self.assertEqual(a_1.annotations['epoch_type'], trials_1_2.annotations['epoch_type'])
        assert_arrays_equal(trials_1_2.array_annotations['trial_id'], np.array([1, 2]))
        self.assertIsInstance(trials_1_2.array_annotations, ArrayDict)

        # test getting more than one event time of more than one event
        trials_1_2b = get_epochs(block, trial_id=[1, 2])

        self.assertEqual(len(trials_1_2b), 2)

        a_idx = np.where(np.array([ev.annotations['epoch_type']
                                   for ev in trials_1_2b]) == 'a')[0][0]

        trials_1_2b_a = trials_1_2b[a_idx]
        trials_1_2b_b = trials_1_2b[a_idx - 1]

        assert_same_attributes(trials_1_2b_a, trials_1_2)

        self.assertEqual(b_1.name, trials_1_2b_b.name)
        self.assertEqual(b_1.description, trials_1_2b_b.description)
        self.assertEqual(b_1.file_origin, trials_1_2b_b.file_origin)
        self.assertEqual(b_1.annotations['epoch_type'], trials_1_2b_b.annotations['epoch_type'])
        assert_arrays_equal(trials_1_2b_b.array_annotations['trial_id'], np.array([1, 2]))
        self.assertIsInstance(trials_1_2b_b.array_annotations, ArrayDict)

    def test__add_epoch(self):
        starts = Event(times=[0.5, 10.0, 25.2] * pq.s)
        starts.annotate(event_type='trial start', nix_name='neo.event.0')
        starts.array_annotate(trial_id=[1, 2, 3])

        stops = Event(times=[5.5, 14.9, 30.1] * pq.s)
        stops.annotate(event_type='trial stop', nix_name='neo.event.1')
        stops.array_annotate(trial_id=[1, 2, 3])

        seg = Segment()
        seg.events = [starts, stops]

        # test cutting with one event only
        ep_starts = add_epoch(seg, starts, pre=-300 * pq.ms, post=250 * pq.ms)

        assert_neo_object_is_compliant(ep_starts)
        self.assertDictEqual(ep_starts.annotations, {'event_type': 'trial start'})
        assert_arrays_almost_equal(ep_starts.times, starts.times - 300 * pq.ms, 1e-12)
        assert_arrays_almost_equal(ep_starts.durations,
                                   (550 * pq.ms).rescale(ep_starts.durations.units)
                                   * np.ones(len(starts)), 1e-12)

        # test cutting with two events
        ep_trials = add_epoch(seg, starts, stops)

        assert_neo_object_is_compliant(ep_trials)
        self.assertDictEqual(ep_trials.annotations, {'event_type': 'trial start'})
        assert_arrays_almost_equal(ep_trials.times, starts.times, 1e-12)
        assert_arrays_almost_equal(ep_trials.durations, stops - starts, 1e-12)

    def test__match_events(self):
        starts = Event(times=[0.5, 10.0, 25.2] * pq.s)
        starts.annotate(event_type='trial start')
        starts.array_annotate(trial_id=[1, 2, 3])

        stops = Event(times=[5.5, 14.9, 30.1] * pq.s)
        stops.annotate(event_type='trial stop')
        stops.array_annotate(trial_id=[1, 2, 3])

        stops2 = Event(times=[0.1, 5.5, 5.6, 14.9, 25.2, 30.1] * pq.s)
        stops2.annotate(event_type='trial stop')
        stops2.array_annotate(trial_id=[1, 1, 2, 2, 3, 3])

        # test for matching input events, should just return identical copies
        matched_starts, matched_stops = match_events(starts, stops)

        assert_same_attributes(matched_starts, starts)
        assert_same_attributes(matched_stops, stops)

        # test for non-matching input events, should find shortest positive non-zero durations
        matched_starts2, matched_stops2 = match_events(starts, stops2)

        assert_same_attributes(matched_starts2, starts)
        assert_same_attributes(matched_stops2, stops)

    def test__cut_block_by_epochs(self):
        epoch = Epoch([0.5, 10.0, 25.2] * pq.s, durations=[5.1, 4.8, 5.0] * pq.s,
                      t_start=.1 * pq.s)
        epoch.annotate(epoch_type='a', pick='me', nix_name='neo.epoch.0')
        epoch.array_annotate(trial_id=[1, 2, 3])

        epoch2 = Epoch([0.6, 9.5, 16.8, 34.1] * pq.s, durations=[4.5, 4.8, 5.0, 5.0] * pq.s,
                       t_start=.1 * pq.s)
        epoch2.annotate(epoch_type='b', nix_name='neo.epoch.1')
        epoch2.array_annotate(trial_id=[1, 2, 3, 4])

        event = Event(times=[0.5, 10.0, 25.2] * pq.s, t_start=.1 * pq.s)
        event.annotate(event_type='trial start', nix_name='neo.event.0')
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

        # test without resetting the time
        seg = Segment(nix_name='neo.segment.0')
        seg2 = Segment(name='NoCut', nix_name='neo.segment.1')
        seg.epochs = [epoch, epoch2]
        seg.events = [event]
        seg.analogsignals = [anasig]
        seg.irregularlysampledsignals = [irrsig]
        seg.spiketrains = [st]

        original_block = Block()
        original_block.segments = [seg, seg2]
        original_block.create_many_to_one_relationship()

        with warnings.catch_warnings(record=True) as w:
            # This should raise a warning as one segment does not contain epochs
            block = cut_block_by_epochs(original_block, properties={'pick': 'me'})
            self.assertEqual(len(w), 1)

        assert_neo_object_is_compliant(block)
        self.assertEqual(len(block.segments), 3)

        for epoch_idx in range(len(epoch)):
            self.assertEqual(len(block.segments[epoch_idx].events), 1)
            self.assertEqual(len(block.segments[epoch_idx].spiketrains), 1)
            self.assertEqual(len(block.segments[epoch_idx].analogsignals), 1)
            self.assertEqual(len(block.segments[epoch_idx].irregularlysampledsignals), 1)

            annos = block.segments[epoch_idx].annotations
            # new segment objects have different identity
            self.assertNotIn('nix_name', annos)

            if epoch_idx != 0:
                self.assertEqual(len(block.segments[epoch_idx].epochs), 1)
            else:
                self.assertEqual(len(block.segments[epoch_idx].epochs), 2)

            assert_same_attributes(block.segments[epoch_idx].spiketrains[0],
                                   st.time_slice(t_start=epoch.times[epoch_idx],
                                                 t_stop=epoch.times[epoch_idx]
                                                        + epoch.durations[epoch_idx]))
            assert_same_attributes(block.segments[epoch_idx].analogsignals[0],
                                   anasig.time_slice(t_start=epoch.times[epoch_idx],
                                                     t_stop=epoch.times[epoch_idx]
                                                            + epoch.durations[epoch_idx]))
            assert_same_attributes(block.segments[epoch_idx].irregularlysampledsignals[0],
                                   irrsig.time_slice(t_start=epoch.times[epoch_idx],
                                                     t_stop=epoch.times[epoch_idx]
                                                            + epoch.durations[epoch_idx]))
            assert_same_attributes(block.segments[epoch_idx].events[0],
                                   event.time_slice(t_start=epoch.times[epoch_idx],
                                                    t_stop=epoch.times[epoch_idx]
                                                           + epoch.durations[epoch_idx]))
        assert_same_attributes(block.segments[0].epochs[0],
                               epoch.time_slice(t_start=epoch.times[0],
                                                 t_stop=epoch.times[0] + epoch.durations[0]))
        assert_same_attributes(block.segments[0].epochs[1],
                               epoch2.time_slice(t_start=epoch.times[0],
                                                t_stop=epoch.times[0] + epoch.durations[0]))

        # test with resetting the time
        seg = Segment(nix_name='neo.segment.0')
        seg2 = Segment(name='NoCut', nix_name='neo.segment.1')
        seg.epochs = [epoch, epoch2]
        seg.events = [event]
        seg.analogsignals = [anasig]
        seg.irregularlysampledsignals = [irrsig]
        seg.spiketrains = [st]

        original_block = Block()
        original_block.segments = [seg, seg2]
        original_block.create_many_to_one_relationship()

        with warnings.catch_warnings(record=True) as w:
            # This should raise a warning as one segment does not contain epochs
            block = cut_block_by_epochs(original_block, properties={'pick': 'me'}, reset_time=True)
            self.assertEqual(len(w), 1)

        assert_neo_object_is_compliant(block)
        self.assertEqual(len(block.segments), 3)

        for epoch_idx in range(len(epoch)):
            self.assertEqual(len(block.segments[epoch_idx].events), 1)
            self.assertEqual(len(block.segments[epoch_idx].spiketrains), 1)
            self.assertEqual(len(block.segments[epoch_idx].analogsignals), 1)
            self.assertEqual(len(block.segments[epoch_idx].irregularlysampledsignals), 1)

            annos = block.segments[epoch_idx].annotations
            self.assertNotIn('nix_name', annos)

            if epoch_idx != 0:
                self.assertEqual(len(block.segments[epoch_idx].epochs), 1)
            else:
                self.assertEqual(len(block.segments[epoch_idx].epochs), 2)

            assert_same_attributes(block.segments[epoch_idx].spiketrains[0],
                                   st.time_shift(- epoch.times[epoch_idx]).time_slice(
                                       t_start=0 * pq.s, t_stop=epoch.durations[epoch_idx]))

            anasig_target = anasig.time_shift(- epoch.times[epoch_idx])
            anasig_target = anasig_target.time_slice(t_start=0 * pq.s,
                                                     t_stop=epoch.durations[epoch_idx])
            assert_same_attributes(block.segments[epoch_idx].analogsignals[0], anasig_target)
            irrsig_target = irrsig.time_shift(- epoch.times[epoch_idx])
            irrsig_target = irrsig_target.time_slice(t_start=0 * pq.s,
                                                     t_stop=epoch.durations[epoch_idx])
            assert_same_attributes(block.segments[epoch_idx].irregularlysampledsignals[0],
                                   irrsig_target)
            assert_same_attributes(block.segments[epoch_idx].events[0],
                                   event.time_shift(- epoch.times[epoch_idx]).time_slice(
                                       t_start=0 * pq.s, t_stop=epoch.durations[epoch_idx]))

        assert_same_attributes(block.segments[0].epochs[0],
                               epoch.time_shift(- epoch.times[0]).time_slice(t_start=0 * pq.s,
                                                                    t_stop=epoch.durations[0]))
        assert_same_attributes(block.segments[0].epochs[1],
                               epoch2.time_shift(- epoch.times[0]).time_slice(t_start=0 * pq.s,
                                                                    t_stop=epoch.durations[0]))


class TestUtilsWithProxyObjects(BaseProxyTest):
    def test__get_events(self):
        starts_1 = Event(times=[0.5, 10.0, 25.2] * pq.s)
        starts_1.annotate(event_type='trial start', pick='me')
        starts_1.array_annotate(trial_id=[1, 2, 3])

        stops_1 = Event(times=[5.5, 14.9, 30.1] * pq.s)
        stops_1.annotate(event_type='trial stop')
        stops_1.array_annotate(trial_id=[1, 2, 3])

        proxy_event = EventProxy(rawio=self.reader, event_channel_index=0,
                                 block_index=0, seg_index=0)

        proxy_event.annotate(event_type='trial start')

        seg = Segment()
        seg.events = [starts_1, stops_1, proxy_event]

        # test getting multiple events including a proxy
        extracted_starts = get_events(seg, event_type='trial start')

        self.assertEqual(len(extracted_starts), 2)

        # make sure the event is loaded and a neo.Event object is returned
        self.assertTrue(isinstance(extracted_starts[0], Event))
        self.assertTrue(isinstance(extracted_starts[1], Event))

    def test__get_epochs(self):
        a = Epoch([0.5, 10.0, 25.2] * pq.s, durations=[5.1, 4.8, 5.0] * pq.s)
        a.annotate(epoch_type='a', pick='me')
        a.array_annotate(trial_id=[1, 2, 3])

        b = Epoch([5.5, 14.9, 30.1] * pq.s, durations=[4.7, 4.9, 5.2] * pq.s)
        b.annotate(epoch_type='b')
        b.array_annotate(trial_id=[1, 2, 3])

        proxy_epoch = EpochProxy(rawio=self.reader, event_channel_index=1,
                                 block_index=0, seg_index=0)

        proxy_epoch.annotate(epoch_type='a')

        seg = Segment()
        seg.epochs = [a, b, proxy_epoch]

        # test getting multiple epochs including a proxy
        extracted_epochs = get_epochs(seg, epoch_type='a')

        self.assertEqual(len(extracted_epochs), 2)

        # make sure the epoch is loaded and a neo.Epoch object is returned
        self.assertTrue(isinstance(extracted_epochs[0], Epoch))
        self.assertTrue(isinstance(extracted_epochs[1], Epoch))

    def test__add_epoch(self):
        proxy_event = EventProxy(rawio=self.reader, event_channel_index=0,
                                 block_index=0, seg_index=0)

        loaded_event = proxy_event.load()

        regular_event = Event(times=loaded_event.times - 1 * loaded_event.units)

        loaded_event.annotate(nix_name='neo.event.0')
        regular_event.annotate(nix_name='neo.event.1')

        seg = Segment()
        seg.events = [regular_event, proxy_event]

        # test cutting with two events one of which is a proxy
        epoch = add_epoch(seg, regular_event, proxy_event)

        assert_neo_object_is_compliant(epoch)
        exp_annos = {k: v for k, v in regular_event.annotations.items() if k != 'nix_name'}
        self.assertDictEqual(epoch.annotations, exp_annos)
        assert_arrays_almost_equal(epoch.times, regular_event.times, 1e-12)
        assert_arrays_almost_equal(epoch.durations,
                                   np.ones(regular_event.shape) * loaded_event.units, 1e-12)

    def test__match_events(self):
        proxy_event = EventProxy(rawio=self.reader, event_channel_index=0,
                                 block_index=0, seg_index=0)

        loaded_event = proxy_event.load()

        regular_event = Event(times=loaded_event.times - 1 * loaded_event.units,
                              labels=np.array(['trigger_a', 'trigger_b'] * 3, dtype='U12'))

        seg = Segment()
        seg.events = [regular_event, proxy_event]

        # test matching two events one of which is a proxy
        matched_regular, matched_proxy = match_events(regular_event, proxy_event)

        assert_same_attributes(matched_regular, regular_event)
        assert_same_attributes(matched_proxy, loaded_event)

    def test__cut_block_by_epochs(self):
        seg = Segment()

        proxy_anasig = AnalogSignalProxy(rawio=self.reader,
                                        stream_index=0, inner_stream_channels=None,
                                        block_index=0, seg_index=0)
        seg.analogsignals.append(proxy_anasig)

        proxy_st = SpikeTrainProxy(rawio=self.reader, spike_channel_index=0,
                                     block_index=0, seg_index=0)
        seg.spiketrains.append(proxy_st)

        proxy_event = EventProxy(rawio=self.reader, event_channel_index=0,
                                 block_index=0, seg_index=0)
        seg.events.append(proxy_event)

        proxy_epoch = EpochProxy(rawio=self.reader, event_channel_index=1,
                                 block_index=0, seg_index=0)
        proxy_epoch.annotate(pick='me')
        seg.epochs.append(proxy_epoch)

        loaded_epoch = proxy_epoch.load()
        loaded_event = proxy_event.load()
        loaded_st = proxy_st.load()
        loaded_anasig = proxy_anasig.load()

        original_block = Block()
        original_block.segments = [seg]
        original_block.create_many_to_one_relationship()

        block = cut_block_by_epochs(original_block, properties={'pick': 'me'})

        assert_neo_object_is_compliant(block)
        self.assertEqual(len(block.segments), proxy_epoch.shape[0])

        for epoch_idx in range(len(loaded_epoch)):
            sliced_event = loaded_event.time_slice(t_start=loaded_epoch.times[epoch_idx],
                                                 t_stop=loaded_epoch.times[epoch_idx]
                                                        + loaded_epoch.durations[epoch_idx])
            has_event = len(sliced_event) > 0

            sliced_anasig = loaded_anasig.time_slice(t_start=loaded_epoch.times[epoch_idx],
                                                   t_stop=loaded_epoch.times[epoch_idx]
                                                          + loaded_epoch.durations[epoch_idx])

            sliced_st = loaded_st.time_slice(t_start=loaded_epoch.times[epoch_idx],
                                                   t_stop=loaded_epoch.times[epoch_idx]
                                                          + loaded_epoch.durations[epoch_idx])

            self.assertEqual(len(block.segments[epoch_idx].events), int(has_event))
            self.assertEqual(len(block.segments[epoch_idx].spiketrains), 1)
            self.assertEqual(len(block.segments[epoch_idx].analogsignals), 1)

            self.assertTrue(isinstance(block.segments[epoch_idx].spiketrains[0],
                                       SpikeTrain))
            assert_same_attributes(block.segments[epoch_idx].spiketrains[0],
                                   sliced_st)

            self.assertTrue(isinstance(block.segments[epoch_idx].analogsignals[0],
                                       AnalogSignal))
            assert_same_attributes(block.segments[epoch_idx].analogsignals[0],
                                   sliced_anasig)

            if has_event:
                self.assertTrue(isinstance(block.segments[epoch_idx].events[0],
                                           Event))
                assert_same_attributes(block.segments[epoch_idx].events[0],
                                   sliced_event)

        block2 = Block()
        seg2 = Segment()
        epoch = Epoch(np.arange(10) * pq.s, durations=np.ones(10) * pq.s)
        epoch.annotate(pick='me instead')
        seg2.epochs = [proxy_epoch, epoch]
        block2.segments = [seg2]
        block2.create_many_to_one_relationship()

        # test correct loading and slicing of EpochProxy objects
        # (not tested above since we used the EpochProxy to cut the block)

        block3 = cut_block_by_epochs(block2, properties={'pick': 'me instead'})

        for epoch_idx in range(len(epoch)):
            sliced_epoch = loaded_epoch.time_slice(t_start=epoch.times[epoch_idx],
                                                   t_stop=epoch.times[epoch_idx]
                                                          + epoch.durations[epoch_idx])
            has_epoch = len(sliced_epoch) > 0

            if has_epoch:
                self.assertTrue(isinstance(block3.segments[epoch_idx].epochs[0],
                                           Epoch))
                assert_same_attributes(block3.segments[epoch_idx].epochs[0],
                                       sliced_epoch)


if __name__ == "__main__":
    unittest.main()
