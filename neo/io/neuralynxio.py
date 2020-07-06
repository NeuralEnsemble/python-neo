"""
Class for reading data from Neuralynx files.
This IO supports NCS, NEV and NSE file formats.

Depends on: numpy

Supported: Read

Author: Julia Sprenger, Carlos Canova
"""

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.neuralynxrawio import NeuralynxRawIO


class NeuralynxIO(NeuralynxRawIO, BaseFromRaw):
    """
    Class for reading data from Neuralynx files.
    This IO supports NCS, NEV, NSE and NTT file formats.

    NCS contains signals for one channel
    NEV contains events
    NSE contains spikes and waveforms for mono electrodes
    NTT contains spikes and waveforms for tetrodes
    """
    _prefered_signal_group_mode = 'group-by-same-units'
    mode = 'dir'

    def __init__(self, dirname, use_cache=False, cache_path='same_as_resource',
                 keep_original_times=False):
        """
        Initialise IO instance

        Parameters
        ----------
        dirname : str
            Directory containing data files
        use_cache : bool, optional
            Cache results of initial file scans for faster loading in subsequent runs.
            Default: False
        cache_path : str, optional
            Folder path to use for cache files.
            Default: 'same_as_resource'
        keep_original_times : bool
            Preserve original time stamps as in data files. By default datasets are
            shifted to begin at t_start = 0*pq.second.
            Default: False
        """
        NeuralynxRawIO.__init__(self, dirname=dirname, use_cache=use_cache,
                                cache_path=cache_path, keep_original_times=keep_original_times)
        BaseFromRaw.__init__(self, dirname)
