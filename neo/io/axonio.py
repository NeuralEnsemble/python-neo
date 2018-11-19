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

    Here an important note from erikli@github for user who want to get the :
    With Axon ABF2 files, the information that you need to recapitulate the original stimulus waveform (both digital and analog) is contained in multiple places.

     - `AxonIO._axon_info['protocol']` -- things like number of samples in episode
     - `AxonIO.axon_info['section']['ADCSection']` | `AxonIO.axon_info['section']['DACSection']` -- things about the number of channels and channel properties
     - `AxonIO._axon_info['protocol']['nActiveDACChannel']` -- bitmask specifying which DACs are actually active
     - `AxonIO._axon_info['protocol']['nDigitalEnable']` -- bitmask specifying which set of Epoch timings should be used to specify the duration of digital outputs
    - `AxonIO._axon_info['dictEpochInfoPerDAC']` -- dict of dict. First index is DAC channel and second index is Epoch number (i.e. information about Epoch A in Channel 2 would be in `AxonIO._axon_info['dictEpochInfoPerDAC'][2][0]`)
     - `AxonIO._axon_info['EpochInfo']` -- list of dicts containing information about each Epoch's digital out pattern. Digital out is a bitmask with least significant bit corresponding to Digital Out 0
     - `AxonIO._axon_info['listDACInfo']` -- information about DAC name, scale factor, holding level, etc
     - `AxonIO._t_starts` -- start time of each sweep in a unified time basis
     - `AxonIO._sampling_rate`

    The current AxonIO.read_protocol() method utilizes a subset of these.
    In particular I know it doesn't consider `nDigitalEnable`, `EpochInfo`, or `nActiveDACChannel` and it doesn't account 
    for different types of Epochs offered by Clampex/pClamp other than discrete steps (such as ramp, pulse train, etc and
    encoded by `nEpochType` in the EpochInfoPerDAC section). I'm currently parsing a superset of the properties used 
    by read_protocol() in my analysis scripts, but that code still doesn't parse the full information and isn't in a state
    where it could be committed and I can't currently prioritize putting together all the code that would parse the full
    set of data. The `AxonIO._axon_info['EpochInfo']` section doesn't currently exist.

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
