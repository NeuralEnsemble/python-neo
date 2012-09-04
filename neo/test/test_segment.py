"""
Tests of the Segment class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.core.segment import Segment
from neo.core import RecordingChannelGroup, Unit, Block, SpikeTrain, AnalogSignalArray
import numpy as np
import quantities as pq

class TestSegment(unittest.TestCase):
    def test_init(self):
        seg = Segment(name='a segment', index=3)
        self.assertEqual(seg.name, 'a segment')
        self.assertEqual(seg.file_origin, None)
        self.assertEqual(seg.index, 3)


    def test__construct_subsegment_by_unit(self):
        nb_seg = 3
        nb_unit = 7
        unit_with_sig = [0, 2, 5]
        signal_types = ['Vm', 'Conductances']
        sig_len = 100

        #recordingchannelgroups
        rcgs = [ RecordingChannelGroup(name = 'Vm', channel_indexes = unit_with_sig),
                        RecordingChannelGroup(name = 'Conductance', channel_indexes = unit_with_sig), ]

        # Unit
        all_unit = [ ]
        for u in range(nb_unit):
            un = Unit(name = 'Unit #{}'.format(u), channel_indexes = [u])
            all_unit.append(un)

        bl = Block()
        for s in range(nb_seg):
            seg = Segment(name = 'Simulation {}'.format(s))
            for j in range(nb_unit):
                st = SpikeTrain([1, 2, 3], units = 'ms', t_start = 0., t_stop = 10)
                st.unit = all_unit[j]

            for t in signal_types:
                anasigarr = AnalogSignalArray( np.zeros((sig_len, len(unit_with_sig)) ), units = 'nA',
                                sampling_rate = 1000.*pq.Hz, channel_indexes = unit_with_sig )
                seg.analogsignalarrays.append(anasigarr)

        # what you want
        subseg = seg.construct_subsegment_by_unit(all_unit[:4])


if __name__ == "__main__":
    unittest.main()
