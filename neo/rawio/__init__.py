# -*- coding: utf-8 -*-
"""
:mod:`neo.rawio` provides classes for reading with low level API
electrophysiological data files.


Classes:


.. autoclass:: neo.rawio.BlackrockRawIO

"""

from neo.rawio.axonrawio import AxonRawIO
from neo.rawio.blackrockrawio import BlackrockRawIO
from neo.rawio.brainvisionrawio import BrainVisionRawIO
from neo.rawio.elanrawio import ElanRawIO
from neo.rawio.micromedrawio import MicromedRawIO
from neo.rawio.neuralynxrawio import NeuralynxRawIO
from neo.rawio.neuroexplorerrawio import NeuroExplorerRawIO
from neo.rawio.neuroscoperawio import NeuroScopeRawIO
from neo.rawio.plexonrawio import PlexonRawIO
from neo.rawio.rawbinarysignalrawio import RawBinarySignalRawIO
from neo.rawio.spike2rawio import Spike2RawIO
from neo.rawio.tdtrawio import TdtRawIO
from neo.rawio.winedrrawio import WinEdrRawIO
from neo.rawio.winwcprawio import WinWcpRawIO

rawiolist = [
    AxonRawIO,
    BlackrockRawIO,
    BrainVisionRawIO,
    ElanRawIO,
    MicromedRawIO,
    NeuralynxRawIO,
    NeuroExplorerRawIO,
    NeuroScopeRawIO,
    PlexonRawIO,
    RawBinarySignalRawIO,
    Spike2RawIO,
    TdtRawIO,
    WinEdrRawIO,
    WinWcpRawIO,
]

import os


def get_rawio_class(filename_or_dirname):
    """
    Return a neo.rawio class guess from file extention.
    """
    _, ext = os.path.splitext(filename_or_dirname)
    ext = ext[1:]
    possibles = []
    for rawio in rawiolist:
        if any(ext.lower() == ext2.lower() for ext2 in rawio.extensions):
            possibles.append(rawio)

    if len(possibles) == 1:
        return possibles[0]
    else:
        return None
