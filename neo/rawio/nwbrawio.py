# -*- coding: utf-8 -*-
"""
Class for reading data from a Neurodata Without Borders (NWB) dataset
Documentation : https://neurodatawithoutborders.github.io
Depends on: h5py, nwb, dateutil
Supported: Read, Write
Specification - https://github.com/NeurodataWithoutBorders/specification
Python APIs - (1) https://github.com/AllenInstitute/nwb-api/tree/master/ainwb 
	          (2) https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/core/nwb_data_set.py 
              (3) https://github.com/NeurodataWithoutBorders/api-python
Sample datasets from CRCNS - https://crcns.org/NWB
Sample datasets from Allen Institute - http://alleninstitute.github.io/AllenSDK/cell_types.html#neurodata-without-borders
"""
# neo imports
#from __future__ import unicode_literals # is not compatible with numpy.dtype both py2 py3
from __future__ import print_function, division, absolute_import
#from itertools import chain
#import shutil
from os.path import join
#import dateutil.parser
import quantities as pq
from neo.rawio.baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype, 
        _event_channel_dtype)
from neo.core import (Segment, SpikeTrain, Unit, Epoch, Event, AnalogSignal,
                      IrregularlySampledSignal, ChannelIndex, Block)
from collections import OrderedDict

# Standard Python imports
import tempfile
from tempfile import NamedTemporaryFile
import os
import glob
from scipy.io import loadmat
import numpy as np
from datetime import datetime

# PyNWB imports
import pynwb
# Creating and writing NWB files
from pynwb import NWBFile,TimeSeries, get_manager
from pynwb.base import ProcessingModule
#from pynwb.misc import UnitTimes #, SpikeUnit
##from pynwb.form.backends.hdf5 import HDF5IO
# Creating TimeSeries
from pynwb.ecephys import ElectricalSeries, Device, EventDetection
from pynwb.behavior import SpatialSeries
######from pynwb.epoch import EpochTimeSeries, Epoch ###
from pynwb.image import ImageSeries
from pynwb.core import set_parents
# For Neurodata Type Specifications
from pynwb.spec import NWBAttributeSpec # Attribute Specifications
from pynwb.spec import NWBDatasetSpec # Dataset Specifications
from pynwb.spec import NWBGroupSpec
from pynwb.spec import NWBNamespace
##from pynwb.form.build import GroupBuilder, DatasetBuilder
##from pynwb.form.spec import NamespaceCatalog
#
from pynwb import *

import h5py
from pynwb import get_manager
##from pynwb.form.backends.hdf5 import HDF5IO
##from pynwb.form import *
##from pynwb.form.build.builders import *


class NWBRawIO(BaseRawIO):
    """
    Class for "reading" experimental data from a .nwb file
    """   
    extensions = ['nwb']    
    rawmode = 'one-file'
    
    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        print("filename = ", filename)
        self.filename = filename
        print("self.filename = ", self.filename)
   
    def _source_name(self):
	    return self.filename

    def _parse_header(self):        
        print("******************** def parse*********************************************")
        print("pynwb.__version__ = ", pynwb.__version__)
        
        io = pynwb.NWBHDF5IO(self.filename, mode='r') # Open a file with NWBHDF5IO
        print("io = ", io)
#        io = self.read_builder_NWB()

        self._file = io.read() # Define the file as a NWBFile object
        print("self._file = ", self._file)
        print("   ")

        print("****************************************sig unit channels******************")
        # Definition of signal channels
        sig_channels = []
        # Definition of units channels
        unit_channels = []

        self.header = {}

        #
        # "i" define as an object the kind of signal (TimeSeries, SpatialSeries, ElectricalSeries), or units (SpikeEventSeries).
        # And for each, thank to loops, we can have access to the differents parameters of the signal_channels, as 
        # the channel name, the id channel, the sampling rate, the type, data units, the resolution, the offset, and the group_id.
        #

######## For sig_channels ########

        for i in self._file.acquisition:
            print("----------------------------acquisition-----------------------------1--------------")                       
            print("i = ", i)
       #     print("range(len(self._file.acquisition)) = ", range(len(self._file.acquisition))) ### 
       #     print("len(self._file.acquisition) = ", len(self._file.acquisition)) ###

            print("######## For sig_channels ########")

                # Channnel name
            ch_name = i.name # name of the dataset
            print("ch_name = ", ch_name)

                # id channel
                # index as name               
            chan_id = i.source
            print("chan_id = ", chan_id)

                # sampling rate
            sr = i.rate
            print("sr = ", sr)

                # dtype
            dtype = i.data.dtype
            print("dtype = ", dtype)

                # units of data
            units = i.unit
            print("units = ", units)

                # gain
            gain = i.resolution
            print("gain = ", gain)

                # offset
            offset = 0.
            print("offset = ", offset)
 
                #group_id is only for special cases when channel have diferents sampling rate for instance. 
            group_id = 0
            print("group_id = ", group_id)

            sig_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, group_id))
            print("sig_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, group_id)) = ", sig_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, group_id)))

        sig_channels = np.array(sig_channels)
        print("----------------------------------------------------------------------------------------------------------------------sig_channels = ", sig_channels)


######## For unit_channels ########

        for i in self._file.acquisition:
            print("------------------------------------------------------unit----acquisition---------------------------------------")
            print("i = ", i)

            print("######## For unit_channels ########")

            unit_name = 'unit{}'.format(i.name)
            print("unit_channels = ", unit_channels)

            unit_id = '#{}'.format(i.source)
            print("unit_id = ", unit_id)

            wf_units = i.timestamps_unit
            print("wf_units = ", wf_units)

            wf_gain = i.resolution
            print("wf_gain = ", wf_gain)

            wf_offset = 0.
            print("wf_offset = ", wf_offset)

            wf_left_sweep = 0
            print("wf_left_sweep = ", wf_left_sweep)

            wf_sampling_rate = i.rate
            print("wf_sampling_rate = ", wf_sampling_rate)

            unit_channels.append((unit_name, unit_id, wf_units, wf_gain, wf_offset, wf_left_sweep, wf_sampling_rate))

            unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)
            print("unit_channels = ", unit_channels)



        print("******************************************event channel***********************************************")
        # Creating event/epoch channel
        # In RawIO epoch and event are dealt the same way.
        event_channels = []
        # Note that an NWB Epoch corresponds to a Neo Segment, not to a Neo Epoch
        # For event
        #event_channels.append(('Some events', 'ev_0', 'event'))
        event_channels.append((self._file.epochs, 'ev_0', 'event')) # Some events

        # For epochs
        #event_channels.append(('Some epochs', 'ep_1', 'epoch'))
        event_channels.append((self._file.epochs, 'ep_1', 'epoch')) # Some epochs

        event_channels = np.array(event_channels, dtype=_event_channel_dtype)
        print("***********************event_channels = ", event_channels)

        print("*******************************************************block**********************************************")
        # file into header dict
#        self.header = {}
        self.header['nb_block'] =2 # 1
        self.header['nb_segment'] = [2, 3] # [1]
        


#####################################################################
        #  file into header dict for signal_channels
        self.header['signal_channels'] = sig_channels

#####################################################################
        # file into header dict for unit channels
        self.header['unit_channels'] = unit_channels
        # file into header dict for event channels
        self.header['event_channels'] = event_channels


        # insert some annotation at some place
        # To create an empty tree
        self._generate_minimal_annotations()
        bl_annotations = self.raw_annotations['blocks'][0]
        seg_annotations = bl_annotations['segments'][0]


    def _segment_t_start(self, block_index, seg_index): # NWB Epoch corresponds to a Neo Segment
        print("def _segment_t_start")
        all_starts = [[0., 15.], [0., 20., 60.]]
        return all_starts[block_index][seg_index]
        return all_starts
       
    def _segment_t_stop(self, block_index, seg_index): # NWB Epoch corresponds to a Neo Segment
        print("def _segment_t_stop")
        all_stops = [[10., 25.], [10., 30., 70.]]
       # return all_stops[block_index][seg_index]
        return all_stops






# ###################################
# # A copy of the end of baserawio.py

    ###
    # signal and channel zone
    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        print("*** _get_signal_size ***")
#        raise (NotImplementedError)
        for i in self._file.acquisition:
            signal_size = i.num_samples
            print("signal_size = ", signal_size) # Same as _spike_count ?
        return signal_size

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        print("*** _get_signal_t_start ***")
#        raise (NotImplementedError)
        for i in self._file.acquisition:
            starting_time = i.starting_time
            print("starting_time = ", starting_time) # For TimeSeries
        return starting_time

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        print("*** _get_analogsignal_chunk ***")
#        raise (NotImplementedError)

    ###
    # spiketrain and unit zone
    def _spike_count(self, block_index, seg_index, unit_index):
        print("*** _spike_count ***")
        #raise (NotImplementedError)
        for i in self._file.acquisition:
            nb_spikes = i.num_samples
        print("nb_spikes = ", nb_spikes)
        return nb_spikes

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        print("*** _get_spike_timestamps ***")
        #raise (NotImplementedError)
        for i in self._file.acquisition: 
            spike_timestamps = i.timestamps
        print("spike_timestamps = ", spike_timestamps)
        return spike_timestamps


    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        print("*** _rescale_spike_timestamp ***")
        #raise (NotImplementedError)
        for i in self._file.acquisition:
            spike_times = spike_timestamps.astype(dtype)
            spike_times /= i.sr
        print("spike_times = ", spike_times)
        return spike_times

    ###
    # spike waveforms zone
    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        print("*** _get_spike_raw_waveforms ***")
        raise (NotImplementedError)

    ###
    # event and epoch zone
    def _event_count(self, block_index, seg_index, event_channel_index):
        print("*** _event_count ***")
        #raise (NotImplementedError)
        for i in self._file.acquisition:
            event_count = i.num_samples
        print("event_count = ", event_count) # Same as nb_spikes ?
        return event_count



    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        print("*** _get_event_timestamps ***")
        raise (NotImplementedError)

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        print("*** _rescale_event_timestamp ***")
        raise (NotImplementedError)

    def _rescale_epoch_duration(self, raw_duration, dtype):
        print("*** _rescale_epoch_duration ***")
        raise (NotImplementedError)



