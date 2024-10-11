"""
IO for reading MED datasets using dhn-med-py library.

dhn-med-py
https://medformat.org
https://pypi.org/project/dhn-med-py/

MED Format Specifications: https://medformat.org

Author: Dan Crepeau, Matt Stead
"""

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.medrawio import MedRawIO


class MedIO(MedRawIO, BaseFromRaw):
    """
    IO for reading MED datasets.
    """

    name = "MED IO"
    description = "IO for reading MED datasets"

    _prefered_signal_group_mode = "group-by-same-units"
    mode = "dir"

    def __init__(self, dirname=None, password=None, keep_original_times=False):
        MedRawIO.__init__(self, dirname=dirname, password=password, keep_original_times=keep_original_times)
        """
        Initialise IO instance

        Parameters
        ----------
        dirname : str
            Directory containing data files
        password : str
            MED sessions can be optionally encrypted with a password.
            Default: None
        keep_original_times : bool
            Preserve original time stamps as in data files. By default datasets are
            shifted to begin at t_start = 0.  When set to True, timestamps will be
            returned as UTC (seconds since midnight 1 Jan 1970).
            Default: False
        """
        BaseFromRaw.__init__(self, dirname)

    def close(self):
        MedRawIO.close(self)

    def __del__(self):
        MedRawIO.__del__(self)
