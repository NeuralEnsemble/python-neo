"""
NWBIO
========

IO class for reading data from a Neurodata Without Borders (NWB) dataset

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

from __future__ import absolute_import
from __future__ import division
from itertools import chain
import shutil
import tempfile
from datetime import datetime
from os.path import join
import dateutil.parser
import numpy as np

import quantities as pq
from neo.io.baseio import BaseIO
from neo.core import (Segment, SpikeTrain, Unit, Epoch, Event, AnalogSignal,
                      IrregularlySampledSignal, ChannelIndex, Block)

# neo imports
from collections import OrderedDict

# Standard Python imports
from tempfile import NamedTemporaryFile
import os
import glob
from scipy.io import loadmat

# PyNWB imports
import pynwb
from pynwb import *
from pynwb import NWBFile,TimeSeries, get_manager
from pynwb.base import ProcessingModule
from pynwb.ecephys import ElectricalSeries, Device, EventDetection
from pynwb.behavior import SpatialSeries
from pynwb.image import ImageSeries
from pynwb.core import set_parents
from pynwb.spec import NWBAttributeSpec # Attribute Specifications
from pynwb.spec import NWBDatasetSpec # Dataset Specifications
from pynwb.spec import NWBGroupSpec
from pynwb.spec import NWBNamespace

# allensdk package
import allensdk
from allensdk import *
from pynwb import load_namespaces
from allensdk.brain_observatory.nwb.metadata import load_LabMetaData_extension
from allensdk.brain_observatory.behavior.schemas import OphysBehaviorMetaDataSchema, OphysBehaviorTaskParametersSchema
load_LabMetaData_extension(OphysBehaviorMetaDataSchema, 'AIBS_ophys_behavior')
load_LabMetaData_extension(OphysBehaviorTaskParametersSchema, 'AIBS_ophys_behavior')


class NWBIO(BaseIO):
    """
    Class for "reading" experimental data from a .nwb file.
    """

    is_readable = True
    is_writable = True
    is_streameable = False
    supported_objects = [Block, Segment, AnalogSignal, IrregularlySampledSignal,
                         SpikeTrain, Epoch, Event]
    readable_objects  = supported_objects
    writeable_objects = supported_objects
    has_header = False
    name = 'NWB'
    description = 'This IO reads/writes experimental data from/to an .nwb dataset'
    extensions = ['nwb']
    mode = 'one-file'

    def __init__(self, filename):
        """
        Arguments:
            filename : the filename
        """
        BaseIO.__init__(self, filename=filename)
        self.filename = filename
        io = pynwb.NWBHDF5IO(self.filename, mode='r') # Open a file with NWBHDF5IO
        self._file = io.read() # Define the file as a NWBFile object

    def read_block(self, lazy=False, cascade=True, **kwargs):
        self._lazy = lazy
        file_access_dates = self._file.file_create_date
        identifier = self._file.identifier
        if identifier == '_neo':  # this is an automatically generated name used if block.name is None
            identifier = None
        description = self._file.session_description
        if description == "no description":
            description = None
        block = Block(name=identifier, 
                      description=description,
                      file_origin=self.filename,
                      file_datetime=file_access_dates,
                      rec_datetime=self._file.session_start_time,
                      file_access_dates=file_access_dates,
                      file_read_log='')
        if cascade:
            self._handle_general_group(block)
            self._handle_epochs_group(lazy, block)
            self._handle_acquisition_group(lazy, block)
            self._handle_stimulus_group(lazy, block)
            self._handle_processing_group(block)
            self._handle_analysis_group(block)
        self._lazy = False
        return block

    def write_block(self, block, **kwargs):
        start_time = datetime.now()
        for i in self._file.acquisition:
            data = self._file.get_acquisition(i).data
            unit = self._file.get_acquisition(i).unit
            name = self._file.get_acquisition(i).name
            comments = self._file.get_acquisition(i).comments
            timestamps = self._file.get_acquisition(i).rate
            start_time = self._file.get_acquisition(i).starting_time
        nwb_timeseries = TimeSeries(name=name, data=data, unit=unit, timestamps=[timestamps])
        nwb_epoch = self._file.add_epoch(start_time, 4.0, [comments], [nwb_timeseries, ]) ### Check 4.0 !
        segments = self._file.epochs[0]
        block = self.read_block(block)
        block.segments.append(block)
        for segment in block.segments:
            self._write_segment(segment)

    def _handle_general_group(self, block):
        pass

    def _handle_epochs_group(self, lazy, block):
        # Note that an NWB Epoch corresponds to a Neo Segment, not to a Neo Epoch.
        self._lazy = lazy
        epochs = self._file.acquisition
        for key in epochs:    
            timeseries = []
            current_shape = self._file.get_acquisition(key).data.shape[0]
            times = np.zeros(current_shape)
            for j in range(0, current_shape):
                times[j]=1./self._file.get_acquisition(key).rate*j+self._file.get_acquisition(key).starting_time
                if times[j] == self._file.get_acquisition(key).starting_time:
                    t_start = times[j] * pq.second
                elif times[j]==times[-1]:
                    t_stop = times[j] * pq.second
                else:
                    timeseries.append(self._handle_timeseries(self._lazy, key, times[j]))
                segment = Segment(name=j)
            for obj in timeseries:
                obj.segment = segment
                if isinstance(obj, AnalogSignal):
                    segment.analogsignals.append(obj)
                elif isinstance(obj, IrregularlySampledSignal):
                    segment.irregularlysampledsignals.append(obj)
                elif isinstance(obj, Event):
                    segment.events.append(obj)
                elif isinstance(obj, Epoch):
                    segment.epochs.append(obj)
            segment.block = block
            segment.times=times
            return segment, obj, times

    def _handle_timeseries(self, lazy, name, timeseries):
        for i in self._file.acquisition:
            data_group = self._file.get_acquisition(i).data*self._file.get_acquisition(i).conversion
            dtype = data_group.dtype
            if lazy==True:
                data = np.array((), dtype=dtype)
                lazy_shape = data_group.shape
            else:
                data = data_group

            if dtype.type is np.string_:
                if self._lazy:
                    times = np.array(())
                else:
                    times = self._file.get_acquisition(i).timestamps
                duration = 1/self._file.get_acquisition(i).rate
                if durations:
                    # Epoch
                    if self._lazy:
                        durations = np.array(())
                    obj = Epoch(times=times,
                                durations=durations,
                                labels=data_group,
                                units='second')
                else:
                    # Event
                    obj = Event(times=times,
                                labels=data_group,
                                units='second')
            else:
                units = self._file.get_acquisition(i).unit
            current_shape = self._file.get_acquisition(i).data.shape[0] # number of samples
            times = np.zeros(current_shape)
            for j in range(0, current_shape):
                times[j]=1./self._file.get_acquisition(i).rate*j+self._file.get_acquisition(i).starting_time
                if times[j] == self._file.get_acquisition(i).starting_time:
                    sampling_metadata = times[j]
                    t_start = sampling_metadata * pq.s
                    sampling_rate = self._file.get_acquisition(i).rate * pq.Hz
                    obj = AnalogSignal(
                                       data_group,
                                       units=units,
                                       sampling_rate=sampling_rate,
                                       t_start=t_start,
                                       name=name)
                elif self._file.get_acquisition(i).timestamps:
                    if self._lazy:
                        time_data = np.array(())
                    else:
                        time_data = self._file.get_acquisition(i).timestamps
                    obj = IrregularlySampledSignal(
                                                data_group,
                                                units=units,
                                                time_units=pq.second)
            return obj


    def _handle_acquisition_group(self, lazy, block):
        acq = self._file.acquisition

    def _handle_stimulus_group(self, lazy, block):
        sti = self._file.stimulus
        for name in sti:
            segment_name_sti = self._file.epochs
            desc_sti = self._file.get_stimulus(name).unit
            segment_sti = segment_name_sti
            if lazy==True:
                times = np.array(())
                lazy_shape = self._file.get_stimulus(name).data.shape
            else:
                current_shape = self._file.get_stimulus(name).data.shape[0] # sample number
                times = np.zeros(current_shape)
                for j in range(0, current_shape): # For testing !
                    times[j]=1./self._file.get_stimulus(name).rate*j+self._file.get_acquisition(name).starting_time # times = 1./frequency [Hz] + t_start [s]
                spiketrain = SpikeTrain(times, units=pq.second,
                                         t_stop=times[-1]*pq.second)

    def _handle_processing_group(self, block):
        pass

    def _handle_analysis_group(self, block):
        pass

    def _write_segment(self, segment):
        for i in self._file.acquisition:
            name = i
            data = self._file.get_acquisition(i).data
            unit = self._file.get_acquisition(i).unit
            name = self._file.get_acquisition(i).name
            comments = self._file.get_acquisition(i).comments
            timestamps = self._file.get_acquisition(i).rate
            start_time = self._file.get_acquisition(i).starting_time
            rate = self._file.get_acquisition(i).rate
            num_samples = self._file.get_acquisition(i).num_samples
            starting_time_unit = self._file.get_acquisition(i).starting_time_unit
            timestamps_unit = self._file.get_acquisition(i).timestamps_unit

        nwb_timeseries = TimeSeries(name=name, data=data, unit=unit, timestamps=[timestamps])       
        nwb_epoch = self._file.add_epoch(start_time, 4.0, [comments], [nwb_timeseries, ]) ### Check 4.0 !

        # AnalogSignal
        segment = Segment(num_samples)
        sig0 = AnalogSignal(signal=data[:], units=unit, sampling_rate=rate*pq.Hz)
        segment.analogsignals.append(sig0)

        # SpikeTrain
#        stop_times=self._file.epochs[0][2]
#        train0 = SpikeTrain(times= nwb_timeseries.data[:], units='sec', t_stop=stop_times)
#        segment.spiketrains.append(train0)

        for i, signal in enumerate(chain(segment.analogsignals, segment.irregularlysampledsignals)):
            self._write_signal(signal, nwb_epoch, i)
        self._write_spiketrains(self._handle_acquisition_group, self._handle_epochs_group(True, Block)[0])
        for i, event in enumerate(segment.events):
            self._write_event(event, nwb_epoch, i)
        for i, neo_epoch in enumerate(segment.epochs):
            self._write_neo_epoch(neo_epoch, nwb_epoch, i)

    def _write_signal(self, signal, epoch, i):
        for i in self._file.acquisition:
            name = i
        signal_name = signal.name or "signal{0}".format(i)
        ts_name = "{0}".format(signal_name)        

        for i in self._file.acquisition:
            ts = self._file.get_acquisition(i).data[:]

        conversion = _decompose_unit(signal.units)
        attributes = {"conversion": conversion,
                      "resolution": float('nan')}

        if isinstance(signal, AnalogSignal):
            sampling_rate = signal.sampling_rate.rescale("Hz")
            signal.sampling_rate = sampling_rate
        else:
            raise TypeError("signal has type {0}, should be AnalogSignal or IrregularlySampledSignal".format(signal.__class__.__name__))

    def _write_spiketrains(self, spiketrains, segment):
        pass

    def _write_event(self, event, nwb_epoch, i):
        event_name = event.name or "event{0}".format(i)
        ts_name = "{0}_{1}".format(event.segment.name, event_name)
        self._file.add_epoch_ts(
                               nwb_epoch,
                               time_in_seconds(event.segment.t_start),
                               time_in_seconds(event.segment.t_stop),
                               event_name,
                                )

    def _write_neo_epoch(self, neo_epoch, nwb_epoch, i):
        pass

def time_in_seconds(t):
    return float(t.rescale("second"))

def _decompose_unit(unit):
    assert isinstance(unit, pq.quantity.Quantity)
    assert unit.magnitude == 1
    conversion = 1.0
    def _decompose(unit):
        dim = unit.dimensionality
        if len(dim) != 1:
            raise NotImplementedError("Compound units not yet supported")
        uq, n = dim.items()[0]
        if n != 1:
            raise NotImplementedError("Compound units not yet supported")
        uq_def = uq.definition
        return float(uq_def.magnitude), uq_def

prefix_map = {
    1e-3: 'milli',
    1e-6: 'micro',
    1e-9: 'nano'
}