# encoding: utf-8
"""
Module for reading/writing data from/to PyNN NumpyBinaryFile format.

PyNN is available at http://neuralensemble.org/PyNN

Classes:
    PyNNNumpyIO
    PyNNTextIO

Supported: Read/Write

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


class BasePyNNIO(BaseIO):
    """
    Base class for PyNN IO classes
    """
    is_readable = True 
    is_writable = True
    has_header = True
    is_streameable = False # TODO - correct spelling to "is_streamable"
    supported_objects = [Segment, AnalogSignal, AnalogSignalArray, SpikeTrain]
    readable_objects = supported_objects
    writeable_objects = supported_objects
    mode = 'file'
    
    def _read_file_contents(self):
        raise NotImplementedError
    
    def _extract_array(self, data, channel_index):
        idx = numpy.where(data[:, 1] == channel_index)[0]
        return data[idx, 0]
    
    def _extract_signal(self, data, metadata, channel_index):
        arr = self._extract_array(data, channel_index)
        if len(arr) > 0:
            signal = AnalogSignal(arr,
                                  units=UNITS_MAP[metadata['variable']],
                                  sampling_period=metadata['dt']*pq.ms)
            signal.annotate(label=metadata["label"],
                            variable=metadata["variable"],
                            channel_index=channel_index)
            return signal
    
    def _extract_spikes(self, data, metadata, channel_index):
        spike_times = self._extract_array(data, channel_index)
        if len(spike_times) > 0:
            spiketrain = SpikeTrain(spike_times, units=pq.ms)
            spiketrain.annotate(label=metadata["label"],
                                channel_index=channel_index,
                                dt=metadata["dt"])
            return spiketrain
    
    def _write_file_contents(self, data, metadata):
        raise NotImplementedError
    
    def read(self, **kwargs):
        """
        Read the file.
        Return a neo.Segment
        See read_segment for detail.
        """
        return self.read_segment(**kwargs)
    
    def read_segment(self, lazy=False, cascade=True):
        data, metadata = self._read_file_contents()
        annotations = dict((k, metadata[k]) for k in ("label", "variable", "first_id", "last_id"))
        seg = Segment(**annotations)
        if metadata['variable'] == 'spikes':
            for i in range(metadata['first_index'], metadata['last_index']):
                spiketrain = self._extract_spikes(data, metadata, i)
                if spiketrain is not None:
                    seg._spiketrains.append(spiketrain)
                seg.annotate(dt=metadata['dt']) # store dt for SpikeTrains only, as can be retrieved from sampling_period for AnalogSignal
        else:
            for i in range(metadata['first_index'], metadata['last_index']):
                # probably slow. Replace with numpy-based version from 0.1
                signal = self._extract_signal(data, metadata, i)
                if signal is not None:
                    seg._analogsignals.append(signal)
        return seg

    def write_segment(self, segment):
        source = segment._analogsignals or segment._spiketrains
        assert len(source) > 0, "Segment contains neither analog signals nor spike trains."
        metadata = segment._annotations.copy()
        metadata['size'] = len(source)
        metadata['first_index'] = 0
        metadata['last_index'] = metadata['size']
        s0 = source[0]
        if 'dt' not in metadata: # dt not included in annotations if Segment contains only AnalogSignals
            metadata['dt'] = s0.sampling_period.rescale(pq.ms).magnitude
        n = sum(s.size for s in source)
        metadata['n'] = n
        data = numpy.empty((n, 2))
        for i, signal in enumerate(source): # here signal may be AnalogSignal or SpikeTrain
            data[i*s0.size:(i+1)*s0.size, 0] = numpy.array(signal.rescale(UNITS_MAP[segment.variable]))
            data[i*s0.size:(i+1)*s0.size, 1] = i*numpy.ones((s0.size,), dtype=float) # index
        self._write_file_contents(data, metadata)

    def read_analogsignal(self, lazy=False, channel_index=0): # channel_index should be positional arg, no?
        data, metadata = self._read_file_contents()
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
        data, metadata = self._read_file_contents()
        if metadata['variable'] != 'spikes':
            raise TypeError("File contains analog signals, not spike data")
        else:
            spiketrain = self._extract_spikes(data, metadata, channel_index)
            if spiketrain is None:
                raise IndexError("File does not contain any spikes with channel index %d" % channel_index)
            else:
                return spiketrain


class PyNNNumpyIO(BasePyNNIO):
    """
    Reads/writes data from/to PyNN NumpyBinaryFile format
    """
    name = "PyNN NumpyBinaryFile"
    extensions = ['npz']

    def _read_file_contents(self):
        contents = numpy.load(self.filename)
        data = contents["data"]
        metadata = {}
        for name,value in contents['metadata']:
            try:
                metadata[name] = eval(value)
            except Exception:
                metadata[name] = value
        return data, metadata

    def _write_file_contents(self, data, metadata):
        metadata_array = numpy.array(sorted(metadata.items()))
        numpy.savez(self.filename, data=data, metadata=metadata_array)


class PyNNTextIO(BasePyNNIO):
    """
    Reads/writes data from/to PyNN StandardTextFile format
    """
    name = "PyNN StandardTextFile"
    extensions = ['txt', 'v', 'ras']

    def _read_metadata(self):
        metadata = {}
        with open(self.filename) as f:
            for line in f:
                if line[0] == "#":
                    name, value = line[1:].strip().split("=")
                    name = name.strip()
                    try:
                        metadata[name] = eval(value)
                    except Exception:
                        metadata[name] = value.strip()
                else:
                    break
        return metadata

    def _read_file_contents(self):
        data = numpy.loadtxt(self.filename)
        metadata = self._read_metadata()
        return data, metadata

    def _write_file_contents(self, data, metadata):
        with open(self.filename, 'wb') as f:
            for item in sorted(metadata.items()):
                f.write(("# %s = %s\n" % item).encode('utf8'))
            numpy.savetxt(f, data)
