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
    TdtRawIO,
    WinEdrRawIO,
    WinWcpRawIO,
    
]