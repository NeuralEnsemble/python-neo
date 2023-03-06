"""
:mod:`neo.rawio` provides classes for reading with low level API
electrophysiological data files.

:attr:`neo.rawio.rawiolist` provides a list of successfully imported rawio
classes.

Functions:

.. autofunction:: neo.rawio.get_rawio_class


Classes:

* :attr:`AlphaOmegaRawIO`
* :attr:`AxographRawIO`
* :attr:`AxonaRawIO`
* :attr:`AxonRawIO`
* :attr:`BiocamRawIO`
* :attr:`BlackrockRawIO`
* :attr:`BrainVisionRawIO`
* :attr:`CedRawIO`
* :attr: `EDFRawIO`
* :attr:`ElanRawIO`
* :attr:`IntanRawIO`
* :attr:`MaxwellRawIO`
* :attr:`MEArecRawIO`
* :attr:`MicromedRawIO`
* :attr:`NeuralynxRawIO`
* :attr:`NeuroExplorerRawIO`
* :attr:`NeuroScopeRawIO`
* :attr:`NIXRawIO`
* :attr:`OpenEphysRawIO`
* :attr:`OpenEphysBinaryRawIO`
* :attr:`PhyRawIO`
* :attr:`PlexonRawIO`
* :attr:`RawBinarySignalRawIO`
* :attr:`RawMCSRawIO`
* :attr:`Spike2RawIO`
* :attr:`SpikeGadgetsRawIO`
* :attr:`SpikeGLXRawIO`
* :attr:`TdtRawIO`
* :attr:`WinEdrRawIO`
* :attr:`WinWcpRawIO`


.. autoclass:: neo.rawio.AlphaOmegaRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.AxographRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.AxonaRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.AxonRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.BiocamRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.BlackrockRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.BrainVisionRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.CedRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.EDFRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.ElanRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.IntanRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.MaxwellRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.MEArecRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.MicromedRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.NeuralynxRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.NeuroExplorerRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.NeuroScopeRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.NIXRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.OpenEphysRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.OpenEphysBinaryRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.PhyRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.PlexonRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.RawBinarySignalRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.RawMCSRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.Spike2RawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.SpikeGadgetsRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.SpikeGLXRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.TdtRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.WinEdrRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.WinWcpRawIO

    .. autoattribute:: extensions

"""
import os

from neo.rawio.alphaomegarawio import AlphaOmegaRawIO
from neo.rawio.axographrawio import AxographRawIO
from neo.rawio.axonarawio import AxonaRawIO
from neo.rawio.axonrawio import AxonRawIO
from neo.rawio.biocamrawio import BiocamRawIO
from neo.rawio.blackrockrawio import BlackrockRawIO
from neo.rawio.brainvisionrawio import BrainVisionRawIO
from neo.rawio.cedrawio import CedRawIO
from neo.rawio.edfrawio import EDFRawIO
from neo.rawio.elanrawio import ElanRawIO
from neo.rawio.examplerawio import ExampleRawIO
from neo.rawio.intanrawio import IntanRawIO
from neo.rawio.maxwellrawio import MaxwellRawIO
from neo.rawio.mearecrawio import MEArecRawIO
from neo.rawio.micromedrawio import MicromedRawIO
from neo.rawio.neuralynxrawio import NeuralynxRawIO
from neo.rawio.neuroexplorerrawio import NeuroExplorerRawIO
from neo.rawio.neuroscoperawio import NeuroScopeRawIO
from neo.rawio.nixrawio import NIXRawIO
from neo.rawio.openephysrawio import OpenEphysRawIO
from neo.rawio.openephysbinaryrawio import OpenEphysBinaryRawIO
from neo.rawio.phyrawio import PhyRawIO
from neo.rawio.plexonrawio import PlexonRawIO
from neo.rawio.rawbinarysignalrawio import RawBinarySignalRawIO
from neo.rawio.rawmcsrawio import RawMCSRawIO
from neo.rawio.spike2rawio import Spike2RawIO
from neo.rawio.spikegadgetsrawio import SpikeGadgetsRawIO
from neo.rawio.spikeglxrawio import SpikeGLXRawIO
from neo.rawio.tdtrawio import TdtRawIO
from neo.rawio.winedrrawio import WinEdrRawIO
from neo.rawio.winwcprawio import WinWcpRawIO

rawiolist = [
    AlphaOmegaRawIO,
    AxographRawIO,
    AxonaRawIO,
    AxonRawIO,
    BiocamRawIO,
    BlackrockRawIO,
    BrainVisionRawIO,
    CedRawIO,
    EDFRawIO,
    ElanRawIO,
    IntanRawIO,
    MicromedRawIO,
    MaxwellRawIO,
    MEArecRawIO,
    NeuralynxRawIO,
    NeuroExplorerRawIO,
    NeuroScopeRawIO,
    NIXRawIO,
    OpenEphysRawIO,
    OpenEphysBinaryRawIO,
    PhyRawIO,
    PlexonRawIO,
    RawBinarySignalRawIO,
    RawMCSRawIO,
    Spike2RawIO,
    SpikeGadgetsRawIO,
    SpikeGLXRawIO,
    TdtRawIO,
    WinEdrRawIO,
    WinWcpRawIO,
]


def get_rawio_class(filename_or_dirname):
    """
    Return a neo.rawio class guess from file extension.
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
