"""
Tests of neo.io.neomatlabio
"""

import unittest

import quantities as pq
from neo import Block, Segment, SpikeTrain
from neo.test.iotest.common_io_test import BaseTestIO
from neo.io.neomatlabio import NeoMatlabIO, HAVE_SCIPY


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
        block1.segments.append(seg)
        seg.spiketrains.append(spiketrain1)

        # write block
        filename = BaseTestIO.get_filename_path(self, 'matlabiotestfile.mat')
        io1 = self.ioclass(filename)
        io1.write_block(block1)

        # read block
        io2 = self.ioclass(filename)
        block2 = io2.read_block()

        self.assertEqual(block1.segments[0].spiketrains[0],
                         block2.segments[0].spiketrains[0])

        # test annotations
        spiketrain2 = block2.segments[0].spiketrains[0]
        assert 'yep' in spiketrain2.annotations
        assert spiketrain2.annotations['yep'] == 'yop'


if __name__ == "__main__":
    unittest.main()
