"""
Reading from neuroscope format files.
Ref: http://neuroscope.sourceforge.net/

It is an old format from Buzsaki's lab

Some old open datasets from spike sorting
are still using this format.

This only the signals.
This should be done (but maybe never will):
  * SpikeTrain file   '.clu'  '.res'
  * Event  '.ext.evt'  or '.evt.ext'

Author: Samuel Garcia

"""
from pathlib import Path

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                        _spike_channel_dtype, _event_channel_dtype)

import numpy as np
from xml.etree import ElementTree


class NeuroScopeRawIO(BaseRawIO):
    extensions = ['xml', 'dat', 'lfp', 'eeg']
    rawmode = 'one-file'

    def __init__(self, filename, binary_file=None):
        """raw reader for Neuroscope

        Parameters
        ----------
        filename : str, Path
            Usually the path of an xml file
        binary_file : str or Path optional
            The binary data file
            Supported formats: ['.dat', '.lfp', '.eeg']
            
        Neuroscope format is composed of two files: a xml file with metadata and a binary file
        in either .dat, .lfp or .eeg format.
        
        For backwards compatibility, we offer three ways of initializing the reader.
            
        Cases:
        filename provided with .xml extension:
            - If binary_file is provided, it is used as the data file.
            - If binary_file is not provided, it tries to find a binary file with the same name and the
            supported extensions (.dat, .lfp, .eeg) in that order.
        filename provided with empty extension:
            - If binary_file is provided, it is used as the data file.
            - If binary_file is not provided, it tries to find a binary file with the same name and the
            supported extensions (.dat, .lfp, .eeg) in that order.
        filename provided with a supported data extension (.dat, .lfp, .eeg):
            - It assumes that the XML file has the same name and a .xml extension.
        """
        BaseRawIO.__init__(self)
        self.filename = filename
        self.binary_file = binary_file

    def _source_name(self):
        return Path(self.filename).stem

    def _parse_header(self):
        # Load the right paths to xml and data
        self._resolve_xml_and_data_paths()

        # Parse XML-file
        tree = ElementTree.parse(self.xml_file_path)
        root = tree.getroot()
        acq = root.find('acquisitionSystem')
        nbits = int(acq.find('nBits').text)
        nb_channel = int(acq.find('nChannels').text)
        self._sampling_rate = float(acq.find('samplingRate').text)
        voltage_range = float(acq.find('voltageRange').text)
        # offset = int(acq.find('offset').text)
        amplification = float(acq.find('amplification').text)

        # find groups for channels
        channel_group = {}
        for grp_index, xml_chx in enumerate(
                root.find('anatomicalDescription').find('channelGroups').findall('group')):
            for xml_rc in xml_chx:
                channel_group[int(xml_rc.text)] = grp_index

        if nbits == 16:
            sig_dtype = 'int16'
            gain = voltage_range / (2 ** 16) / amplification / 1000.
            # ~ elif nbits==32:
            # Not sure if it is int or float
            # ~ dt = 'int32'
            # ~ gain  = voltage_range/2**32/amplification
        else:
            raise (NotImplementedError)

        # Extract signal from the data file
        self._raw_signals = np.memmap(self.data_file_path, dtype=sig_dtype,
                                      mode='r', offset=0).reshape(-1, nb_channel)

        # one unique stream
        signal_streams = np.array([('Signals', '0')], dtype=_signal_stream_dtype)

        # signals
        sig_channels = []
        for c in range(nb_channel):
            name = 'ch{}grp{}'.format(c, channel_group.get(c, 'none'))
            chan_id = str(c)
            units = 'mV'
            offset = 0.
            stream_id = '0'
            sig_channels.append((name, chan_id, self._sampling_rate,
                                 sig_dtype, units, gain, offset, stream_id))
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = sig_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._raw_signals.shape[0] / self._sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        assert stream_index == 0
        return self._raw_signals.shape[0]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        assert stream_index == 0
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        if channel_indexes is None:
            channel_indexes = slice(None)
        raw_signals = self._raw_signals[slice(i_start, i_stop), channel_indexes]
        return raw_signals

    def _resolve_xml_and_data_paths(self):
        """
        Resolves XML and data paths from the provided filename and binary_file attributes.
        
        See the __init__ of the class for more a description of the conditions.
        
        Using these conditions these function updates the self.xml_file_path and self.data_file_path attributes.        

        """

        supported_extensions = ['.dat', '.lfp', '.eeg']
        self.filename = Path(self.filename)
        self.binary_file = Path(self.binary_file) if self.binary_file is not None else None
        
        if self.filename.suffix == '.xml':
            xml_file_path = self.filename
            data_file_path = self.binary_file 
        elif self.filename.suffix == '':
            xml_file_path = self.filename.with_suffix(".xml")
            data_file_path = self.binary_file
        elif self.filename.suffix in supported_extensions:
            xml_file_path = self.filename.with_suffix(".xml")
            data_file_path = self.filename
        else:
            raise KeyError(f"Format {self.filename.suffix} not supported, filename format should be {supported_extensions} or .xml")
        
        if data_file_path is None:
            possible_file_paths = (xml_file_path.with_suffix(extension) for extension in supported_extensions)
            data_file_path = next((path for path in possible_file_paths if path.is_file()), None)
            if data_file_path is None:
                raise FileNotFoundError(f"data binary not found for file {xml_file_path.stem} with supported extensions: {supported_extensions}")

        
        assert xml_file_path.is_file(), f"xml file not found at the expected location {xml_file_path}"
        assert data_file_path.is_file(), f"binary file not found at the expected location {data_file_path}"

        self.xml_file_path = xml_file_path
        self.data_file_path = data_file_path