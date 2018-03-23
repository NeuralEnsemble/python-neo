# -*- coding: utf-8 -*-

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.axonrawio import AxonRawIO

from neo.core import Block, Segment, AnalogSignal, Event
import quantities as pq


class AxonIO(AxonRawIO, BaseFromRaw):
    """
    Class for reading data from pCLAMP and AxoScope
    files (.abf version 1 and 2), developed by Molecular device/Axon technologies.

    - abf = Axon binary file
    - atf is a text file based format from axon that could be
      read by AsciiIO (but this file is less efficient.)
    """
    _prefered_signal_group_mode = 'split-all'

    def __init__(self, filename):
        AxonRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)

    def read_protocol(self):
        """
        Read the protocol waveform of the file, if present;
        function works with ABF2 only. Protocols can be reconstructed
        from the ABF1 header.
        Returns: list of segments (one for every episode)
                 with list of analog signls (one for every DAC).
        """
        sigs_by_segments, sig_names, sig_units = self.read_raw_protocol()
        segments = []
        for seg_index, sigs in enumerate(sigs_by_segments):
            seg = Segment(index=seg_index)
            t_start = self._t_starts[seg_index] * pq.s
            for c, sig in enumerate(sigs):
                ana_sig = AnalogSignal(sig, sampling_rate=self._sampling_rate * pq.Hz,
                                       t_start=t_start, name=sig_names[c], units=sig_units[c])
                seg.analogsignals.append(ana_sig)
            segments.append(seg)

        return segments
