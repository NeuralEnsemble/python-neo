# encoding: utf-8
"""
Module for reading data from PyNN NumpyBinaryFile format.

Authors: Andrew Davison, Pierre Yger
"""

from __future__ import with_statement
from .baseio import BaseIO
from ..core import Segment, AnalogSignal, AnalogSignalArray, SpikeTrain
import numpy
import quantities as pq

UNITS_MAP = {
    'spikes': pq.ms,
    'v': pq.mV,
    'gsyn': pq.UnitQuantity('microsiemens', 1e-9*pq.S, 'uS', 'µS'), # check
}

class PyNNBinaryIO(BaseIO):
    """
    
    """
    is_readable = True 
    is_writable = True
    has_header = True
    is_streameable = False # TODO - correct spelling to "is_streamable"
    supported_objects = [Segment, AnalogSignal, AnalogSignalArray, SpikeTrain]
    readable_objects = supported_objects
    writeable_objects = supported_objects
    name = "PyNN NumpyBinaryFile"
    extensions = ['npz']
    mode = 'file'
    
    def read(self, **kwargs):
        """
        Read the file.
        Return a neo.Segment
        See read_segment for detail.
        """
        return self.read_segment(**kwargs)
    
    def _read_arrays(self):
        contents = numpy.load(self.filename)
        data = contents["data"]
        metadata = {}
        for name,value in contents['metadata']:
            try:
                metadata[name] = eval(value)
            except Exception:
                metadata[name] = value
        return data, metadata
    
    def _extract_array(self, data, channel_index):
        idx = numpy.where(data[:, 1] == channel_index)[0]
        return data[idx, 0]
    
    def _extract_signal(self, data, metadata, channel_index):
        signal = self._extract_array(data, channel_index)
        if len(signal) > 0:
            return  AnalogSignal(signal,
                                 units=UNITS_MAP[metadata['variable']],
                                 sampling_period=metadata['dt']*pq.ms)
    
    def _extract_spikes(self, data, metadata, channel_index):
        spike_times = self._extract_array(data, channel_index)
        if len(spike_times) > 0:
            return SpikeTrain(spike_times, units=pq.ms)
    
    def read_segment(self, lazy=False, cascade=True):
        data, metadata = self._read_arrays()
        if metadata['variable'] == 'spikes':
            raise NotImplementedError
        else:
            seg = Segment()
            for i in range(metadata['first_index'], metadata['last_index']):
                signal = self._extract_signal(data, metadata, i)
                if signal is not None:
                    seg._analogsignals.append(signal)
            return seg

    def read_analogsignal(self, lazy=False, channel_index=0): # channel_index should be positional arg, no?
        data, metadata = self._read_arrays()
        if metadata['variable'] == 'spikes':
            raise TypeError("File contains spike data, not analog signals")
        else:
            signal = self._extract_signal(data, metadata, channel_index)
            if signal is None:
                raise IndexError("File does not contain a signal with channel index %d" % channel_index)
            else:
                return signal

    def read_analogsignalarray(self, lazy=False):
        raise NotImplementedError
    
    def read_spiketrain(self, lazy=False, channel_index=0):
        data, metadata = self._read_arrays()
        if metadata['variable'] != 'spikes':
            raise TypeError("File contains analog signals, not spike data")
        else:
            spiketrain = self._extract_spikes(data, metadata, channel_index)
            if spiketrain is None:
                raise IndexError("File does not contain any spikes with channel index %d" % channel_index)
            else:
                return spiketrain
