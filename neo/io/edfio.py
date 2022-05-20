"""
IO for reading edf and edf+ files using pyedflib

PyEDFLib
https://pyedflib.readthedocs.io
https://github.com/holgern/pyedflib

EDF Format Specifications: https://www.edfplus.info/

Author: Julia Sprenger
"""

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.edfrawio import EDFRawIO


class EDFIO(EDFRawIO, BaseFromRaw):
    """
    IO for reading edf and edf+ files.
    """
    name = 'EDF IO'
    description = "IO for reading EDF and EDF+ files"

    _prefered_signal_group_mode = 'group-by-same-units'

    def __init__(self, filename=''):
        EDFRawIO.__init__(self, filename=filename)
        BaseFromRaw.__init__(self, filename)
