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
    
    def read_segment(self, lazy=False, cascade=True):
        with open(self.filename, 'r') as f:
            contents = numpy.load(f)
            data = contents["data"]
            metadata = {}
            for name,value in contents['metadata']:
                try:
                    metadata[name] = eval(value)
                except Exception:
                    metadata[name] = value
        if metadata['variable'] == 'spikes':
            raise NotImplementedError
        else:
            seg = Segment()
            for i in range(metadata['first_index'], metadata['last_index']):
                idx    = numpy.where(data[:, 1] == i)[0]
                signal = data[idx, 0]
                if len(signal) > 0:
                    seg._analogsignals.append(
                        AnalogSignal(signal,
                                     units=UNITS_MAP[metadata['variable']],
                                     sampling_period=metadata['dt']*pq.ms))
            return seg
