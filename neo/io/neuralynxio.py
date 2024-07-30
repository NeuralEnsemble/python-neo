"""
Class for reading data from Neuralynx files.
This IO supports NCS, NEV and NSE file formats.

Depends on: numpy

Supported: Read

Author: Julia Sprenger, Carlos Canova
"""

import warnings
from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.neuralynxrawio.neuralynxrawio import NeuralynxRawIO


class NeuralynxIO(NeuralynxRawIO, BaseFromRaw):
    """
    Class for reading data from Neuralynx files.
    This IO supports NCS, NEV, NSE and NTT file formats.

    NCS contains signals for one channel
    NEV contains events
    NSE contains spikes and waveforms for mono electrodes
    NTT contains spikes and waveforms for tetrodes
    """

    _prefered_signal_group_mode = "group-by-same-units"
    mode = "dir"

    def __init__(
        self,
        dirname,
        use_cache=False,
        cache_path="same_as_resource",
        include_filenames=None,
        exclude_filenames=None,
        keep_original_times=False,
        filename=None,
        exclude_filename=None,
    ):
        """
        Initialise IO instance

        Parameters
        ----------
        dirname : str
            Directory containing data files
        filename : str
            Deprecated and will be removed. Please use `include_filenames` instead
            Name of a single ncs, nse, nev, or ntt file to include in dataset. Will be ignored,
            if dirname is provided.
        use_cache : bool, optional
            Cache results of initial file scans for faster loading in subsequent runs.
            Default: False
        cache_path : str, optional
            Folder path to use for cache files.
            Default: 'same_as_resource'
        exclude_filename: None,
            Deprecated and will be removed. Please use `exclude_filenames` instead
        include_filenames: str or list
            Filename or list of filenames to be included. This can be absolute path or path relative to dirname.
        exclude_filenames: str or list
            Filename or list of filenames to be excluded. Expects base filenames without
            directory path.
        keep_original_times : bool
            Preserve original time stamps as in data files. By default datasets are
            shifted to begin at t_start = 0*pq.second.
            Default: False
        """

        if filename is not None:
            warnings.warn("Deprecated and will be removed. Please use `include_filenames` instead")
            include_filenames = [filename]

        if exclude_filename is not None:
            warnings.warn("Deprecated and will be removed. Please use `exclude_filenames` instead")
            exclude_filenames = exclude_filename

        NeuralynxRawIO.__init__(
            self,
            dirname=dirname,
            include_filenames=include_filenames,
            exclude_filenames=exclude_filenames,
            keep_original_times=keep_original_times,
            use_cache=use_cache,
            cache_path=cache_path,
        )

        if self.rawmode == "one-dir":
            BaseFromRaw.__init__(self, dirname)
        elif self.rawmode == "multiple-files":
            BaseFromRaw.__init__(self, include_filenames=include_filenames)
