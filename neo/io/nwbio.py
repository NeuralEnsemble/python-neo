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
from pynwb.spec import NWBAttributeSpec, NWBDatasetSpec, NWBGroupSpec, NWBNamespace, NWBNamespaceBuilder

# hdmf imports
from hdmf.spec import LinkSpec, GroupSpec, DatasetSpec, SpecNamespace,\
                       NamespaceBuilder, AttributeSpec, DtypeSpec, RefSpec
from hdmf import *


class NWBIO(BaseIO):
    """
    Class for "reading" experimental data from a .nwb file, and "writing" a .nwb file from Neo
    """
    supported_objects = [Block, Segment, AnalogSignal, IrregularlySampledSignal,
                         SpikeTrain, Epoch, Event]
    readable_objects  = supported_objects
    writeable_objects = supported_objects

    has_header = False

    name = 'NeoNWB IO'
    description = 'This IO reads/writes experimental data from/to an .nwb dataset'
    extensions = ['nwb']
    mode = 'one-file'

    is_readable = True
    is_writable = True
    is_streameable = False    

    def __init__(self, filename, mode):
        """
        Arguments:
            filename : the filename
        """
        BaseIO.__init__(self, filename=filename)
        self.filename = filename
   
    def read_all_blocks(self, lazy=False, **kwargs):        
        """
        Loads all blocks in the file that are attached to the root.
        Here, we assume that a neo block is a sub-part of a branch, into a NWB file;
        """
        io = pynwb.NWBHDF5IO(self.filename, mode='r') # Open a file with NWBHDF5IO
        self._file = io.read()
        
        blocks = []
        for node in self._file.acquisition:
            print("node = ", node)
            blocks.append(self._read_block(self._file, node, blocks))            
        return blocks

    def read_block(self, lazy=False, **kargs):
        """
        Load the first block in the file.
        """
        return self.read_all_blocks(lazy=lazy)[0]

    def _read_block(self, _file, node, blocks, lazy=False, cascade=True, **kwargs):   
        """
        Main method to load a block
        """
        self._lazy = lazy

        file_access_dates = _file.file_create_date
        identifier = _file.identifier
        if identifier == '_neo':
            identifier = None
        description = _file.session_description
        if description == "no description":
            description = None

        block = Block(name=identifier,
                      description=description,
                      file_origin=self.filename,
                      file_datetime=file_access_dates,
                      rec_datetime=_file.session_start_time,
                      file_access_dates=file_access_dates,
                      file_read_log='')
        if cascade:
            self._handle_general_group(block)
            self._handle_epochs_group(_file, block)
            self._handle_acquisition_group(lazy, _file, block)
            self._handle_stimulus_group(lazy, _file, block)
            self._handle_processing_group(block)
            self._handle_analysis_group(block)
        self._lazy = False
        return block

    def write_all_blocks(self, blocks, **kwargs):
        """
        Write list of blocks to the file
        """
        start_time = datetime.now()
        nwbfile = NWBFile(self.filename,
                               session_start_time=start_time,
                               identifier='',
                               file_create_date=None,
                               timestamps_reference_time=None,
                               experimenter=None,
                               experiment_description=None,
                               session_id=None,
                               institution=None,
                               keywords=None,
                               notes=None,
                               pharmacology=None,
                               protocol=None,
                               related_publications=None,
                               slices=None,
                               source_script=None,
                               source_script_file_name=None,
                               data_collection=None,
                               surgery=None,
                               virus=None,
                               stimulus_notes=None,
                               lab=None,
                               acquisition=None,
                               stimulus=None,
                               stimulus_template=None,
                               epochs=None,
                               epoch_tags=set(),
                               trials=None,
                               invalid_times=None,
                               units=None,
                               electrodes=None,
                               electrode_groups=None,
                               ic_electrodes=None,
                               sweep_table=None,
                               imaging_planes=None,
                               ogen_sites=None,
                               devices=None,
                               subject=None
                               )
        io_nwb = pynwb.NWBHDF5IO(self.filename, manager=get_manager(), mode='w')

        if Block in self.writeable_objects:
            for block in blocks:
                print("block in all_blocks = ", block)
                self.write_block(nwbfile, block)
                io_nwb.write(nwbfile)
            return list(block.segments)
        io_nwb.close()

    def write_block(self, nwbfile, block, **kwargs):
        """
        Write a Block to the file
            :param block: Block to be written
        """
        self._write_block_children(nwbfile, block)

    def _write_block_children(self, nwbfile, block=None, **kwargs):
        for segment in block.segments:
            print("segment.name = ", segment.name)
            self._write_segment(nwbfile, segment)

    def _handle_general_group(self, block):
        pass

    def _handle_epochs_group(self, _file, block):
        """
        Note that an NWB Epoch corresponds to a Neo Segment, not to a Neo Epoch.
        """
        epochs = _file.epochs
        timeseries=[]
        if epochs is not None:
            t_start = epochs[0][1] * pq.second
            t_stop = epochs[0][2] * pq.second
        segment = Segment(name=self.name)

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
        block.segments.append(segment)

    def _handle_acquisition_group(self, lazy, _file, block):
        acq = _file.acquisition

    def _handle_stimulus_group(self, lazy, _file, block):
        sti = _file.stimulus
        for name in sti:
            segment_name_sti = _file.epochs
            desc_sti = _file.get_stimulus(name).unit
            segment_sti = segment_name_sti
            if lazy==True:
                times = np.array(())
                lazy_shape = _file.get_stimulus(name).data.shape
            else:
                current_shape = _file.get_stimulus(name).data.shape[0]
                times = np.zeros(current_shape)
                for j in range(0, current_shape):
                    times[j]=1./_file.get_stimulus(name).rate*j+_file.get_acquisition(name).starting_time # times = 1./frequency [Hz] + t_start [s]
                spiketrain = SpikeTrain(times, units=pq.second,
                                         t_stop=times[-1]*pq.second)

    def _handle_processing_group(self, block):
        pass

    def _handle_analysis_group(self, block):
        pass

    def _write_segment(self, nwbfile, segment):
        start_time = segment.t_start
        stop_time = segment.t_stop

        for i, signal in enumerate(chain(segment.analogsignals, segment.irregularlysampledsignals)):
            self._write_signal(nwbfile, signal, i, segment)
            tS_seg = TimeSeries(
                        name=segment.name,
                        data=signal,
                        timestamps=[1],
                        description="",
                        )
        self._write_spiketrains(nwbfile, segment.spiketrains, segment)        
        for i, event in enumerate(segment.events):
            self._write_event(nwbfile, event, nwb_epoch, i)
        for i, neo_epoch in enumerate(segment.epochs):
            self._write_neo_epoch(nwbfile, neo_epoch, nwb_epoch, i)
        nwbfile.add_acquisition(tS_seg)

    def _write_signal(self, nwbfile, signal, i, segment):
        signal_name = signal.name or "signal%d" % i
        print("signal_name = ", signal_name)
        ts_name = "{0}".format(signal_name)

        conversion = _decompose_unit(signal.units)
        attributes = {"conversion": conversion,
                      "resolution": float('nan')}

        if isinstance(signal, AnalogSignal):
            sampling_rate = signal.sampling_rate.rescale("Hz")
            signal.sampling_rate = sampling_rate
            # All signals should go in /acquisition
            tS = TimeSeries(name=ts_name, starting_time=time_in_seconds(signal.t_start), data=segment.analogsignals, rate=float(sampling_rate))
            #ts = nwbfile.add_acquisition(tS)
            return [segment.analogsignals]
        elif isinstance(signal, IrregularlySampledSignal):
            tS = TimeSeries(name=ts_name, starting_time=time_in_seconds(signal.t_start), data=signal, timestamps=signal.times.rescale('second').magnitude)
            return [segment.irregularlysampledsignals]
        else:
            raise TypeError("signal has type {0}, should be AnalogSignal or IrregularlySampledSignal".format(signal.__class__.__name__))
        #ts = nwbfile.add_acquisition(tS)

    def _write_spiketrains(self, nwbfile, spiketrains, segment):
        fmt = 'unit_{{0:0{0}d}}_{1}'.format(len(str(len(spiketrains))), segment.name)
        for i, spiketrain in enumerate(spiketrains):
            unit = fmt.format(i)
            ug = nwbfile.add_unit(
                                   spike_times=spiketrain.rescale('second').magnitude,
                                   Modules='',
                                   UnitTimes='',
                                  )

    def _write_event(self, nwbfile, event, nwb_epoch, i):
        event_name = event.name or "event{0}".format(i)
        ts_name = "{0}".format(event_name)
        tS = TimeSeries(
                        name=ts_name,
                        data=event, 
                        timestamps=event.times.rescale('second').magnitude,
                        description=event.description or "",
                        )

        nwbfile.add_epoch(nwb_epoch,     
                          start_time=time_in_seconds(event.times[0]),
                          stop_time=time_in_seconds(event.times[1]),
                          )

    def _write_neo_epoch(self, nwbfile, neo_epoch, nwb_epoch, i):
        neo_epoch_name = neo_epoch.name or "intervalseries{0}".format(i)
        ts_name = "{0}".format(neo_epoch_name)
        tS = TimeSeries(
                        name=ts_name,
                        data=neo_epoch,
                        timestamps=neo_epoch.times.rescale('second').magnitude,
                        description=neo_epoch.description or "",
                        )

        nwbfile.add_epoch(          
                             nwb_epoch,
                             start_time=time_in_seconds(neo_epoch.times[0]),
                             stop_time=time_in_seconds(neo_epoch.times[-1]),
                             )

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