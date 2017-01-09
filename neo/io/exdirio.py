# -*- coding: utf-8 -*-
"""
This is the implementation of the NEO IO for the exdir format.
Depends on: scipy
            h5py >= 2.5.0
            numpy
            quantities
Supported: Read
Authors: Milad H. Mobarhan @CINPLA,
         Svenn-Arne Dragly @CINPLA,
         Mikkel E. Lepperød @CINPLA
"""

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import sys
from neo.io.baseio import BaseIO
from neo.core import (Segment, SpikeTrain, Unit, Epoch, AnalogSignal,
                      ChannelIndex, Block, IrregularlySampledSignal)
import neo.io.tools
import numpy as np
import quantities as pq
import os
import glob
import exdir

python_version = sys.version_info.major
if python_version == 2:
    from future.builtins import str


class ExdirIO(BaseIO):
    """
    Class for reading/writting of exdir fromat
    """

    is_readable = True
    is_writable = False

    supported_objects = [Block, Segment, AnalogSignal, ChannelIndex, SpikeTrain]
    readable_objects = [Block, SpikeTrain]
    writeable_objects = []

    has_header = False
    is_streameable = False

    name = 'exdir'
    description = 'This IO reads experimental data from an eds folder'

    # mode can be 'file' or 'dir' or 'fake' or 'database'
    # the main case is 'file' but some reader are base on a directory or a database
    # this info is for GUI stuff also
    mode = 'dir'

    def __init__(self, folder_path):
        """
        Arguments:
            folder_path : the folder path
        """
        BaseIO.__init__(self)
        self._absolute_folder_path = folder_path
        self._path, relative_folder_path = os.path.split(folder_path)
        self._base_folder, extension = os.path.splitext(relative_folder_path)

        if extension != ".exdir":
            raise ValueError("folder extension must be '.exdir'")

        self._exdir_folder = exdir.File(folder=folder_path, mode="a")
        
        # TODO check if group exists
        self._processing = self._exdir_folder.require_group("processing")

    def read_block(self,
                   lazy=False,
                   cascade=True):
        # TODO read block
        blk = Block()
        if cascade:
            seg = Segment(file_origin=self._absolute_folder_path)

            for name in self._processing:
                if(name == "Position"):
                    seg.irregularlysampledsignals += self.read_tracking(path="")
                if(name == "LFP"):
                    seg.analogsignals += self.read_analogsignal(path="")
                if(name == "EventWaveform"):
                    seg.spiketrains += self.read_spiketrain(path="")
                    
                for key in self._processing[name]:
                    if(key == "Position"):
                        seg.irregularlysampledsignals += self.read_tracking(path=name)
                    if(key == "LFP"):
                        seg.analogsignals += self.read_analogsignal(path=name)
                    if(key == "EventWaveform"):
                        seg.spiketrains += self.read_spiketrain(path=name)
                        
            blk.segments += [seg]

            # TODO add duration
            # TODO May need to "populate_RecordingChannel"

        return blk
    
    def read_analogsignal(self, path):
        if(len(path) == 0):
            lfp_group = self._processing["LFP"]
        else:
            lfp_group = self._processing[path]["LFP"]
            
        analogsignals = []
        
        for key in lfp_group:
            timeserie = lfp_group[key]
            signal = timeserie["data"]
            analogsignal = AnalogSignal(
                signal.data,
                units=signal.attrs["unit"], 
                sampling_rate=pq.Quantity(
                    timeserie.attrs["sample_rate"]["value"],
                    timeserie.attrs["sample_rate"]["unit"]
                )
            )
            
            analogsignals.append(analogsignal)
            
            # TODO: what about channel index
            # TODO: read attrs?
            
        return analogsignals

    def read_spiketrain(self, path):
        # TODO implement read spike train
        if(len(path) == 0):
            event_waveform_group = self._processing["EventWaveform"]
        else:
            event_waveform_group = self._processing[path]["EventWaveform"]
            
        spike_trains = []
        
        for key in event_waveform_group:
            timeserie = event_waveform_group[key]
            timestamps = timeserie["timestamps"]
            waveforms = timeserie["waveforms"]

            spike_train = SpikeTrain(
                pq.Quantity(timestamps.data, timestamps.attrs["unit"]),
                t_stop=pq.Quantity(
                    timestamps.data[-1], 
                    timestamps.attrs["unit"]
                ),
                waveforms=pq.Quantity(
                    waveforms.data, 
                    waveforms.attrs["unit"]
                )
            )
            
            spike_trains.append(spike_train)
            # TODO: read attrs?
            
        return spike_trains

    def read_epoch(self):
        # TODO read epoch data
        pass
        
    def read_tracking(self, path):
        """
        Read tracking data_end
        """
        if(len(path) == 0):
            pos_group = self._processing["Position"]
        else:
            pos_group = self._processing[path]["Position"]
        irr_signals = []
        for key in pos_group:
            spot_group = pos_group[key]
            times = spot_group["timestamps"]
            coords = spot_group["data"]
            irr_signal = IrregularlySampledSignal(name=pos_group[key].name,
                                                  signal=coords.data,
                                                  times=times.data,
                                                  units=coords.attrs["unit"],
                                                  time_units=times.attrs["unit"])
            irr_signals.append(irr_signal)
        return irr_signals
        

if __name__ == "__main__":
    import sys
    testfile = "/tmp/test.exdir"
    io = ExdirIO(testfile)
    
    block = io.read_block()

    from neo.io.hdf5io import NeoHdf5IO

    testfile = "/tmp/test_exdir_to_neo.h5"
    try:
        os.remove(testfile)
    except:
        pass
    hdf5io = NeoHdf5IO(testfile)
    hdf5io.write(block)
