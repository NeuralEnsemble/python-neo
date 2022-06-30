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

    def __init__(self, filename='', binary_file=None):
        """raw reader for Neuroscope

        Parameters
        ----------
        filename : str, optional
            Usually the file_path of an xml file
        binary_file : _type_, optional
            The binary data file corresponding to the xml file:
            Supported formats: ['.dat', '.lfp', '.eeg']
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
        file_path = Path(self.filename)
        supported_data_extensions = ['.dat', '.lfp', '.eeg']
        suffix = file_path.suffix
        if suffix == '.xml':
            xml_file_path = file_path
            data_file_path = self.binary_file
            # If no binary file provided iterate over the formats
            if data_file_path is None:
                for extension in supported_data_extensions:
                    data_file_path = file_path.with_suffix(extension)
                    if data_file_path.is_file():
                        break
                assert data_file_path.is_file(), "data binary not found for file " \
                f"{data_file_path} with supported extensions: {supported_data_extensions}"
        elif suffix == '':
            xml_file_path = file_path.with_suffix(".xml")
            data_file_path = self.binary_file
            # Empty suffix to keep testing behavior with non-existing
            # file passing for backwards compatibility
            if data_file_path is None:
                data_file_path = file_path.with_suffix(".dat")
        elif suffix in supported_data_extensions:
            xml_file_path = file_path.with_suffix(".xml")
            data_file_path = file_path
        else:
            error_string = (
                f"Format {suffix} not supported "
                f"filename format should be {supported_data_extensions} or .xml"
            )
            raise KeyError(error_string)

        msg = f"xml file not found at the expected location {xml_file_path}"
        assert xml_file_path.is_file(), msg
        msg = f"binary file not found at the expected location {data_file_path}"
        assert data_file_path.is_file(), msg

        self.xml_file_path = xml_file_path
        self.data_file_path = data_file_path
