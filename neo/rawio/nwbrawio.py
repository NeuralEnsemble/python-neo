# -*- coding: utf-8 -*-
"""
NWBRawIO
========

RawIO class for reading data from a Neurodata Without Borders (NWB) dataset

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
from __future__ import print_function, division, absolute_import
from os.path import join
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
from pynwb import *
# Creating and writing NWB files
from pynwb import NWBFile,TimeSeries, get_manager
from pynwb.base import ProcessingModule
# Creating TimeSeries
from pynwb.ecephys import ElectricalSeries, Device, EventDetection
from pynwb.behavior import SpatialSeries
from pynwb.image import ImageSeries
from pynwb.core import set_parents
# For Neurodata Type Specifications
from pynwb.spec import NWBAttributeSpec # Attribute Specifications
from pynwb.spec import NWBDatasetSpec # Dataset Specifications
from pynwb.spec import NWBGroupSpec
from pynwb.spec import NWBNamespace
# Plot the structure of a NWB file
from utils.render import HierarchyDescription, NXGraphHierarchyDescription
from matplotlib import pyplot as plt


class NWBRawIO(BaseRawIO):
    """
    Class for reading experimental data from a .nwb file

    Example:
    >>> import neo
    >>> from neo.rawio import NWBRawIO
    >>> reader = neo.rawio.NWBRawIO(filename)
    >>> reader.parse_header()
    >>> print("reader = ", reader)

    >>> # Plot the structure of the NWB file
    >>> reader.plot()
    """
    name = 'NWBRawIO'
    description = ''
    extensions = ['nwb']
    rawmode = 'one-file'
    
    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename
        io = pynwb.NWBHDF5IO(self.filename, mode='r') # Open a file with NWBHDF5IO
        self._file = io.read() # Define the file as a NWBFile object

    def _source_name(self):
	    return self.filename

    def plot(self, filename=''):
        # Plotting settings
        show_bar_plot = False
        plot_single_file = True
        file_hierarchy = HierarchyDescription.from_hdf5(self.filename)
        file_graph = NXGraphHierarchyDescription(file_hierarchy)
        fig = file_graph.draw(show_plot=False,
                              figsize=(12,11),
                              label_offset=(0.0, 0.0065),
                              label_font_size=10)
        plot_title = filename + ", " + "#Datasets=%i, #Attributes=%i, #Groups=%i, #Links=%i" % (len(file_hierarchy['datasets']), len(file_hierarchy['attributes']), len(file_hierarchy['groups']), len(file_hierarchy['links']))
        plt.title(plot_title)
        plt.savefig('Structure_NWB_File.png')
        plt.show()

    def _parse_header(self):

        sig_channels = [] # Definition of signal channels
        unit_channels = [] # Definition of units channels

        #
        # "i" defines as object the signal type (TimeSeries, SpatialSeries, ElectricalSeries), or units (SpikeEventSeries).
        # And for everyone, thanks to the loops, we can have access to the different parameters of the signal_channels, as
        # the channel name, the id channel, the sampling rate, the type, data units, the resolution, the offset, and the group_id.
        #

        print("self._file.acquisition = ", self._file.acquisition)

######## For sig_channels ########
        for i in range(len(self._file.acquisition)):
            print("----------------------------acquisition------------------------------------------")
            print("i = ", i)
            print("######## For sig_channels ########")

            # Channnel name
            ch_name = 'ch_{}'.format(i)
            ### ch_name = self._file.get_acquisition(i).name
            print("ch_name = ", ch_name)

            # id channel index as name
            chan_id = i + 1
            print("chan_id = ", chan_id)

            for j in self._file.acquisition:
                # sampling rate
                sr = self._file.get_acquisition(j).rate
                print("sr = ", sr)

                # dtype
                # dtype = i.data.dtype
                dtype = 'int'  ###
                print("dtype = ", dtype)

                # units of data
                units = self._file.get_acquisition(j).unit
                print("units = ", units)

                # gain
                gain = self._file.get_acquisition(j).resolution
                print("gain = ", gain)

                # offset
                offset = 0.  ###
                print("offset = ", offset)
 
                #group_id is only for special cases when channel have diferents sampling rate for instance. 
                group_id = 0
                print("group_id = ", group_id)
                print("   ")

        sig_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, group_id))

        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        print("---------------------sig_channels = ", sig_channels)
        print("   ")


######## For unit_channels ########
        for i in self._file.acquisition:
            print("------------------------------------------------------unit----acquisition---------------------------------------")
            print("i = ", i)
            print("######## For unit_channels ########")

            unit_name = 'unit{}'.format(self._file.get_acquisition(i).name)
            print("unit_channels = ", unit_channels)

#            unit_id = '#{}'.format(i.source)
            unit_id = '#{}'
            print("unit_id = ", unit_id)

            wf_units = self._file.get_acquisition(i).timestamps_unit
            print("wf_units = ", wf_units)

            wf_gain = self._file.get_acquisition(i).resolution
            print("wf_gain = ", wf_gain)

            wf_offset = 0.
            print("wf_offset = ", wf_offset)

            wf_left_sweep = 0
            print("wf_left_sweep = ", wf_left_sweep)

            wf_sampling_rate = self._file.get_acquisition(i).rate
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

##        event_channels.append((self._file.epochs[0][3], self._file.epochs[0][0], 'event')) # Some events
#        for j in range(len(self._file.epochs)):
#            print("j = ", j)
#
#            epochs_id = self._file.epochs[j][0]
#            print("epochs_start_id = ", epochs_id)
#   
#            epochs_start_time = self._file.epochs[j][1]
#            print("epochs_start_time = ", epochs_start_time)
#    
#            epochs_stop_time = self._file.epochs[j][2]
#            print("epochs_stop_time = ", epochs_stop_time)
#  
#            epochs_tags = self._file.epochs[j][3]
#            print("epochs_tags = ", epochs_tags)
#        
#        event_channels.append((self._file.epochs[j][3], self._file.epochs[j][0], 'event')) # Some events
# Example
        event_channels = []
        event_channels.append(('Some events', 'ev_0', 'event'))
        event_channels.append(('Some epochs', 'ep_1', 'epoch'))
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # For epochs
        #event_channels.append(('Some epochs', 'ep_1', 'epoch'))
##        event_channels.append((self._file.epochs, 'ep_1', 'epoch')) # Some epochs

#        event_channels = np.array(event_channels, dtype=_event_channel_dtype)
        print("***********************event_channels = ", event_channels)

        print("*******************************************************block**********************************************")
        # file into header dict
        self.header = {}
        self.header['nb_block'] =2 # 1
        self.header['nb_segment'] = [2, 3] # [1]

#####################################################################
        self.header['signal_channels'] = sig_channels #  file into header dict for signal_channels
        self.header['unit_channels'] = unit_channels # file into header dict for unit channels
        self.header['event_channels'] = event_channels # file into header dict for event channels

        # insert some annotation at some place
        # To create an empty tree
        self._generate_minimal_annotations()
#        bl_annotations = self.raw_annotations['blocks'][0]
#        seg_annotations = bl_annotations['segments'][0]


    def _segment_t_start(self, block_index, seg_index): # NWB Epoch corresponds to a Neo Segment
        print("*** def _segment_t_start ***")
        all_starts = [[0., 15.], [0., 20., 60.]]
        return all_starts[block_index][seg_index]
#        for i in self._file.acquisition:
#            print("i = ", i)
#            all_starts = self._file.get_acquisition(i).starting_time
#            print("all_starts = ", all_starts)
#        return np.array(all_starts)
        #return all_starts

       
    def _segment_t_stop(self, block_index, seg_index): # NWB Epoch corresponds to a Neo Segment
        print("*** def _segment_t_stop ***")
        all_stops = [[10., 25.], [10., 30., 70.]]
        return all_stops[block_index][seg_index]
        #return all_stops
#        for i in self._file.acquisition:
#            all_stops = self._file.get_acquisition(i).stop_time
#            print("all_stops = ", all_stops)


# ###################################
# # A copy of the end of baserawio.py

    ###
    # signal and channel zone
    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        print("*** _get_signal_size ***")
        #        raise (NotImplementedError)
##        io = pynwb.NWBHDF5IO(self.filename, mode='r') # Open a file with NWBHDF5IO
##        self._file = io.read()
        for i in self._file.acquisition:
            signal_size = self._file.get_acquisition(i).num_samples
            print("signal_size = ", signal_size) # Same as _spike_count ?
        return signal_size

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        print("*** _get_signal_t_start ***")
#        raise (NotImplementedError)
##        io = pynwb.NWBHDF5IO(self.filename, mode='r') # Open a file with NWBHDF5IO
##        self._file = io.read()
        for i in self._file.acquisition:
            starting_time = self._file.get_acquisition(i).starting_time
#            starting_time =  np.array(starting_time)
        return starting_time

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        print("*** _get_analogsignal_chunk ***")
#        raise (NotImplementedError)
        print("channel_indexes = ", channel_indexes)
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = 100000

        assert i_start >= 0, "I don't like your jokes"
        assert i_stop <= 100000, "I don't like your jokes"

        if channel_indexes is None:
            nb_chan = 16
        else:
            nb_chan = len(channel_indexes)
        raw_signals = np.zeros((i_stop - i_start, nb_chan), dtype='int16')
        return raw_signals

    ###
    # spiketrain and unit zone
    def _spike_count(self, block_index, seg_index, unit_index):
        print("*** _spike_count ***")
        #raise (NotImplementedError)
        for i in self._file.acquisition:
            print("i in _spike_count = ", i)
            nb_spikes = self._file.get_acquisition(i).num_samples
        print("nb_spikes = ", nb_spikes)
        return nb_spikes

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        print("*** _get_spike_timestamps ***")
        #raise (NotImplementedError)
        for i in self._file.acquisition:
            spike_timestamps = self._file.get_acquisition(i).timestamps
            print("spike_timestamps in condition = ", spike_timestamps)
        print("spike_timestamps = ", spike_timestamps)
        return spike_timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        print("*** _rescale_spike_timestamp ***")
        #raise (NotImplementedError)
        for i in self._file.acquisition:
            spike_times = spike_timestamps.astype(dtype)
###            spike_times /= i.sr
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
            event_count = self._file.get_acquisition(i).num_samples
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
