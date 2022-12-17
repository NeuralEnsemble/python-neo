"""
Tests of neo.io.neomatlabio
"""

import unittest
from numpy.testing import assert_array_equal
import quantities as pq

from neo.core.analogsignal import AnalogSignal
from neo.core.irregularlysampledsignal import IrregularlySampledSignal
from neo import Block, Segment, SpikeTrain
from neo.test.iotest.common_io_test import BaseTestIO
from neo.io.neomatlabio import NeoMatlabIO

try:
    import scipy.io
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


@unittest.skipUnless(HAVE_SCIPY, "requires scipy")
class TestNeoMatlabIO(BaseTestIO, unittest.TestCase):
    ioclass = NeoMatlabIO
    files_to_test = []
    files_to_download = []

    def test_write_read_single_spike(self):
        block1 = Block()
        seg = Segment('segment1')
        spiketrain1 = SpikeTrain([1] * pq.s, t_stop=10 * pq.s, sampling_rate=1 * pq.Hz)
        spiketrain1.annotate(yep='yop')
        sig1 = AnalogSignal([4, 5, 6] * pq.A, sampling_period=1 * pq.ms)
        irrsig1 = IrregularlySampledSignal([0, 1, 2] * pq.ms, [4, 5, 6] * pq.A)
        block1.segments.append(seg)
        seg.spiketrains.append(spiketrain1)
        seg.analogsignals.append(sig1)
        seg.irregularlysampledsignals.append(irrsig1)

        # write block
        filename = self.get_local_path('matlabiotestfile.mat')
        io1 = self.ioclass(filename)
        io1.write_block(block1)

        # read block
        io2 = self.ioclass(filename)
        block2 = io2.read_block()

        self.assertEqual(block1.segments[0].spiketrains[0],
                         block2.segments[0].spiketrains[0])

        assert_array_equal(block1.segments[0].analogsignals[0],
                            block2.segments[0].analogsignals[0])

        assert_array_equal(block1.segments[0].irregularlysampledsignals[0].magnitude,
                           block2.segments[0].irregularlysampledsignals[0].magnitude)
        assert_array_equal(block1.segments[0].irregularlysampledsignals[0].times,
                           block2.segments[0].irregularlysampledsignals[0].times)

        # test annotations
        spiketrain2 = block2.segments[0].spiketrains[0]
        assert 'yep' in spiketrain2.annotations
        assert spiketrain2.annotations['yep'] == 'yop'


if __name__ == "__main__":
    unittest.main()
