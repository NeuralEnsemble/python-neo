"""
Tests of neo.io.phyio
"""

import unittest

from neo.io.phyio import PhyIO  # , HAVE_SCIPY
from neo.test.iotest.common_io_test import BaseTestIO
from neo.io.proxyobjects import (AnalogSignalProxy,
                                 SpikeTrainProxy, EventProxy, EpochProxy)
from neo import (AnalogSignal, SpikeTrain)

import quantities as pq
import numpy as np

import tempfile
from pathlib import Path


class TestPhyIO(BaseTestIO, unittest.TestCase):
    ioclass = PhyIO
    entities_to_download = [
        'phy'
    ]
    entities_to_test = [
        'phy/phy_example_0'
    ]

    def test_read_segment_lazy(self):
        dirname = self.get_local_path('phy/phy_example_0')
        r = PhyIO(dirname=dirname)
        seg = r.read_segment(lazy=True)
        for ana in seg.analogsignals:
            assert isinstance(ana, AnalogSignalProxy)
            ana = ana.load()
            assert isinstance(ana, AnalogSignal)
        for st in seg.spiketrains:
            assert isinstance(st, SpikeTrainProxy)
            st = st.load()
            assert isinstance(st, SpikeTrain)

        seg = r.read_segment(lazy=False)
        for anasig in seg.analogsignals:
            assert isinstance(ana, AnalogSignal)
            self.assertNotEqual(anasig.size, 0)
        for st in seg.spiketrains:
            assert isinstance(st, SpikeTrain)
            self.assertNotEqual(st.size, 0)

        # annotations
        assert seg.name == 'Seg #0 Block #0'
        for anasig in seg.analogsignals:
            assert anasig.name is not None
        for st in seg.spiketrains:
            assert st.name is not None
        for ev in seg.events:
            assert ev.name is not None
        for ep in seg.epochs:
            assert ep.name is not None

    def test_read_block(self):
        dirname = self.get_local_path('phy/phy_example_0')
        r = PhyIO(dirname=dirname)
        bl = r.read_block(lazy=True)

    def test_read_segment_with_time_slice(self):
        dirname = self.get_local_path('phy/phy_example_0')
        r = PhyIO(dirname=dirname)
        seg = r.read_segment(time_slice=None)
        spikes_full = seg.spiketrains[0]

        t_start, t_stop = 260 * pq.ms, 1.854 * pq.s
        seg = r.read_segment(time_slice=(t_start, t_stop))
        spikes_slice = seg.spiketrains[0]

        assert spikes_full.size > spikes_slice.size
        assert np.all(spikes_slice >= t_start)
        assert np.all(spikes_slice <= t_stop)
        assert spikes_slice.t_start == t_start
        assert spikes_slice.t_stop == t_stop


if __name__ == "__main__":
    unittest.main()
