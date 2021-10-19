"""
Tests of neo.io.nixio_fr
"""
import numpy as np
import unittest
from quantities import s
from neo.io.nixio_fr import NixIO as NixIOfr
import quantities as pq
from neo.io.nixio import NixIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.core import Block, Segment, AnalogSignal, SpikeTrain, Event

try:
    import nixio as nix

    HAVE_NIX = True
except ImportError:
    HAVE_NIX = False
import os


@unittest.skipUnless(HAVE_NIX, "Requires NIX")
class TestNixfr(BaseTestIO, unittest.TestCase, ):
    ioclass = NixIOfr

    entities_to_download = [
        'nix/nixio_fr.nix'
    ]
    entities_to_test = [
        'nix/nixio_fr.nix'
    ]

    def setUp(self):
        super().setUp()
        self.testfilename = self.get_local_path('nix/nixio_fr.nix')
        self.reader_fr = NixIOfr(filename=self.testfilename)
        self.reader_norm = NixIO(filename=self.testfilename, mode='ro')
        self.blk = self.reader_fr.read_block(block_index=1, load_waveforms=True)
        # read block with NixIOfr
        self.blk1 = self.reader_norm.read_block(index=1)  # read same block with NixIO

    def tearDown(self):
        self.reader_fr.file.close()
        self.reader_norm.close()

    def test_check_same_neo_structure(self):
        self.assertEqual(len(self.blk.segments), len(self.blk1.segments))
        for seg1, seg2 in zip(self.blk.segments, self.blk1.segments):
            self.assertEqual(len(seg1.analogsignals), len(seg2.analogsignals))
            self.assertEqual(len(seg1.spiketrains), len(seg2.spiketrains))
            self.assertEqual(len(seg1.events), len(seg2.events))
            self.assertEqual(len(seg1.epochs), len(seg2.epochs))

    def test_check_same_data_content(self):
        for seg1, seg2 in zip(self.blk.segments, self.blk1.segments):
            for asig1, asig2 in zip(seg1.analogsignals, seg2.analogsignals):
                np.testing.assert_almost_equal(asig1.magnitude, asig2.magnitude)
                # not completely equal
            for st1, st2 in zip(seg1.spiketrains, seg2.spiketrains):
                np.testing.assert_array_equal(st1.magnitude, st2.times)
                for wf1, wf2 in zip(st1.waveforms, st2.waveforms):
                    np.testing.assert_array_equal(wf1.shape, wf2.shape)
                    np.testing.assert_almost_equal(wf1.magnitude, wf2.magnitude)
            for ev1, ev2 in zip(seg1.events, seg2.events):
                np.testing.assert_almost_equal(ev1.times, ev2.times)
                assert np.all(ev1.labels == ev2.labels)
            for ep1, ep2 in zip(seg1.epochs, seg2.epochs):
                assert len(ep1.durations) == len(ep2.times)
                np.testing.assert_almost_equal(ep1.times, ep2.times)
                np.testing.assert_array_equal(ep1.durations, ep2.durations)
                np.testing.assert_array_equal(ep1.labels, ep2.labels)

    def test_analog_signal(self):
        seg1 = self.blk.segments[0]
        an_sig1 = seg1.analogsignals[0]
        assert len(an_sig1) == 30
        an_sig2 = seg1.analogsignals[1]
        assert an_sig2.shape == (50, 3)

    def test_spike_train(self):
        st1 = self.blk.segments[0].spiketrains[0]
        assert np.all(st1.times == np.cumsum(np.arange(0, 1, 0.1)).tolist() * pq.s + 10 * pq.s)

    def test_event(self):
        seg1 = self.blk.segments[0]
        event1 = seg1.events[0]
        raw_time = 10 + np.cumsum(np.array([0, 1, 2, 3, 4]))
        assert np.all(event1.times == np.array(raw_time * pq.s / 1000))
        assert np.all(event1.labels == np.array(['A', 'B', 'C', 'D', 'E'], dtype='U'))
        assert len(seg1.events) == 1

    def test_epoch(self):
        seg1 = self.blk.segments[1]
        seg2 = self.blk1.segments[1]
        epoch1 = seg1.epochs[0]
        epoch2 = seg2.epochs[0]
        assert len(epoch1.durations) == len(epoch1.times)
        assert np.all(epoch1.durations == epoch2.durations)
        assert np.all(epoch1.labels == epoch2.labels)

    def test_annotations(self):
        self.testfilename = self.get_local_path('nix/nixio_fr_ann.nix')
        with NixIO(filename=self.testfilename, mode='ow') as io:
            annotations = {'my_custom_annotation': 'hello block'}
            bl = Block(**annotations)
            annotations = {'something': 'hello hello000'}
            seg = Segment(**annotations)
            an =AnalogSignal([[1, 2, 3], [4, 5, 6]], units='V',
                             sampling_rate=1 * pq.Hz)
            an.annotate(ansigrandom='hello chars')
            an.array_annotate(custom_id=[1, 2, 3])
            sp = SpikeTrain([3, 4, 5]* s, t_stop=10.0)
            sp.annotations['railway'] = 'hello train'
            ev = Event(np.arange(0, 30, 10)*pq.Hz,
                       labels=np.array(['trig0', 'trig1', 'trig2'], dtype='U'))
            ev.annotations['venue'] = 'hello event'
            ev2 = Event(np.arange(0, 30, 10) * pq.Hz,
                       labels=np.array(['trig0', 'trig1', 'trig2'], dtype='U'))
            ev2.annotations['evven'] = 'hello ev'
            seg.spiketrains.append(sp)
            seg.events.append(ev)
            seg.events.append(ev2)
            seg.analogsignals.append(an)
            bl.segments.append(seg)
            io.write_block(bl)
            io.close()

        with NixIOfr(filename=self.testfilename) as frio:
            frbl = frio.read_block()
            assert 'my_custom_annotation' in frbl.annotations
            assert 'something' in frbl.segments[0].annotations
            assert 'ansigrandom' in frbl.segments[0].analogsignals[0].annotations
            assert 'railway' in frbl.segments[0].spiketrains[0].annotations
            assert 'venue' in frbl.segments[0].events[0].annotations
            assert 'evven' in frbl.segments[0].events[1].annotations
            assert 'custom_id' in frbl.segments[0].analogsignals[0].array_annotations
            for anasig in frbl.segments[0].analogsignals:
                for value in anasig.array_annotations.values():
                    assert anasig.shape[-1] == len(value)
        os.remove(self.testfilename)



if __name__ == '__main__':
    unittest.main()
