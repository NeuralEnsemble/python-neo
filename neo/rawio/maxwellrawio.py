"""
Class for reading data from maxwell biosystem device:
  * MaxOne
  * MaxTwo
  
https://www.mxwbio.com/resources/mea/

The implementation is a mix between:
  * the implementation in spikeextractors 
     https://github.com/SpikeInterface/spikeextractors/blob/master/spikeextractors/extractors/maxwellextractors/maxwellextractors.py
 * the implementation in spyking-circus
    https://github.com/spyking-circus/spyking-circus/blob/master/circus/files/maxwell.py



Author : Samuel Garcia
"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                _spike_channel_dtype, _event_channel_dtype)

import numpy as np

try:
    import h5py
    HAVE_H5 = True
except:
    HAVE_H5 = False


class MaxwellRawIO(BaseRawIO):
    """
    Class for reading MaxOne or MaxTwo files.
    """
    extensions = ['h5']
    rawmode = 'one-file'

    def __init__(self, filename='',  rec_name=None):
        BaseRawIO.__init__(self)
        self.filename = filename
        self.rec_name = rec_name

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        try:
            import MEArec as mr
            HAVE_MEAREC = True
        except ImportError:
            HAVE_MEAREC = False
        assert HAVE_H5, 'h5py is not installed'
        
        h5 = h5py.File(self.filename, mode='r')
        self.h5_file = h5
        version = h5['version'][0].decode()
        
        signal_streams = []
        self._signals = {}
        if int(version) == 20160704:
            # one stream only
            #~ sampling_rate = 20000.
            #~ gain_uV = h5['settings']['lsb'][0] * 1e6
            # one segment
            self._signals['well000'] = h5['sig']
            
            signal_streams.append(('well000', 'well000'))
        elif int(version) > 20160704:
            # multi stream stream (one well is one stream)
            
            stream_ids = list(h5['wells'].keys())
            for stream_id in stream_ids:
                rec_names = list(h5['wells'][stream_id].keys())
                if len(rec_names) > 1:
                    if self.rec_name is not None:
                        raise ValueError('several recording need select with rec_name="rec0000"')
                else:
                    self.rec_name = rec_names[0]

                #~ settings = h5['wells'][stream_id][self.rec_name]['settings']
                #~ gain_uV = settings['lsb'][:][0] * 1e6
                signal_streams.append((stream_id, stream_id))
                
                sig_path = h5['wells'][stream_id][self.rec_name]['groups']['routed']['raw']
                self._signals[stream_id] = sig_path
        
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        print(signal_streams)
        
        sig_channels = []
        for stream_id in signal_streams['id']:
            if int(version) == 20160704:
                sr = 20000.
                gain_uV = h5['settings']['lsb'][0] * 1e6
            elif int(version) > 20160704:
                settings = h5['wells'][stream_id][self.rec_name]['settings']
                sr = settings['sampling'][:][0]
                gain_uV = settings['lsb'][:][0] * 1e6
                mapping = settings['mapping']
                channel_ids = np.array(mapping['channel'])
                electrode_ids = np.array(mapping['electrode'])
                mask = channel_ids >= 0
                channel_ids = channel_ids[mask]
                electrode_ids = electrode_ids[mask]
                
            for i, chan_id in enumerate(channel_ids):
                elec_id = electrode_ids[i]
                ch_name = f'ch{chan_id} elec{elec_id}'
                offset_uV = 0
                sig_channels.append((ch_name, str(chan_id), sr, 'uint16', 'uV', gain_uV, offset_uV, stream_id))

        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = sig_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()
        bl_ann = self.raw_annotations['blocks'][0]
        bl_ann['maxwell_version'] = version

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        # TODO
        return 1000.

    def _get_signal_size(self, block_index, seg_index, stream_index):
        stream_id =  self.header['signal_streams'][stream_index]['id']
        sigs = self._signals[stream_id]
        return sigs.shape[1]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):

        stream_id =  self.header['signal_streams'][stream_index]['id']
        sigs = self._signals[stream_id]
        print(sigs)

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = sigs.shape[1]



        if channel_indexes is None:
            channel_indexes = slice(None)
            
        sigs = sigs[channel_indexes, i_start:i_stop]
        sigs =sigs.T
        
        return sigs

