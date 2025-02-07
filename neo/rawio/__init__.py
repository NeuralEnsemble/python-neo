"""
:mod:`neo.rawio` provides classes for reading
electrophysiological data files with a low-level API

:attr:`neo.rawio.rawiolist` provides a list of successfully imported rawio
classes.

Functions:

.. autofunction:: neo.rawio.get_rawio


Classes:

* :attr:`AlphaOmegaRawIO`
* :attr:`AxographRawIO`
* :attr:`AxonaRawIO`
* :attr:`AxonRawIO`
* :attr:`BiocamRawIO`
* :attr:`BlackrockRawIO`
* :attr:`BrainVisionRawIO`
* :attr:`CedRawIO`
* :attr:`EDFRawIO`
* :attr:`ElanRawIO`
* :attr:`IntanRawIO`
* :attr:`MaxwellRawIO`
* :attr:`MedRawIO`
* :attr:`MEArecRawIO`
* :attr:`MicromedRawIO`
* :attr:`NeuralynxRawIO`
* :attr:`NeuroExplorerRawIO`
* :attr:`NeuroNexusRawIO`
* :attr:`NeuroScopeRawIO`
* :attr:`NIXRawIO`
* :attr:`OpenEphysRawIO`
* :attr:`OpenEphysBinaryRawIO`
* :attr:`PhyRawIO`
* :attr:`PlexonRawIO`
* :attr:`Plexon2RawIO`
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

.. autoclass:: neo.rawio.MedRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.MEArecRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.MicromedRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.NeuralynxRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.NeuroExplorerRawIO

    .. autoattribute:: extensions

.. autoclass:: neo.rawio.NeuroNexusRawIO

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

.. autoclass:: neo.rawio.Plexon2RawIO

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

from pathlib import Path
from collections import Counter

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
from neo.rawio.medrawio import MedRawIO
from neo.rawio.micromedrawio import MicromedRawIO
from neo.rawio.neuralynxrawio import NeuralynxRawIO
from neo.rawio.neuroexplorerrawio import NeuroExplorerRawIO
from neo.rawio.neuronexusrawio import NeuroNexusRawIO
from neo.rawio.neuroscoperawio import NeuroScopeRawIO
from neo.rawio.nixrawio import NIXRawIO
from neo.rawio.openephysrawio import OpenEphysRawIO
from neo.rawio.openephysbinaryrawio import OpenEphysBinaryRawIO
from neo.rawio.phyrawio import PhyRawIO
from neo.rawio.plexonrawio import PlexonRawIO
from neo.rawio.plexon2rawio import Plexon2RawIO
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
    MedRawIO,
    NeuralynxRawIO,
    NeuroExplorerRawIO,
    NeuroNexusRawIO,
    NeuroScopeRawIO,
    NIXRawIO,
    OpenEphysRawIO,
    OpenEphysBinaryRawIO,
    PhyRawIO,
    PlexonRawIO,
    Plexon2RawIO,
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
    """Legacy function for returning class guess from file extension
    DEPRECATED
    """

    import warnings

    warnings.warn(
        "get_rawio_class is deprecated and will be removed in 0.15.0. " "In the future please use `get_rawio`"
    )

    return get_rawio(filename_or_dirname)


def get_rawio(filename_or_dirname, exclusive_rawio: bool = True):
    """
    Return a neo.rawio class guess from file extension.

    Parameters
    ----------
    filename_or_dirname : str | Path
        The filename or directory name to check for file suffixes that
        can be read by Neo. This can also be used to check whether a
        rawio could read a not-yet written file
    exclusive_rawio: bool, default: True
        Whether to return a rawio if there is only one rawio capable of
        reading the file. If this doesn't exist will return None.
        If set to False it will return all possible rawios organized
        by the most likely rawio.

    Returns
    -------
    possibles: neo.RawIO | None | list[neo.RawIO]
        If exclusive_rawio is True, returns the single RawIO that
        can read a file/set of files or None. If exclusive_rawio is
        False it will return all possible RawIOs (organized by most likely)
        that could read the file or files.
    """
    filename_or_dirname = Path(filename_or_dirname)

    # if filename_or_dirname doesn't exist then user may just be checking if
    # neo can read their file or they give a real file
    if not filename_or_dirname.exists() or filename_or_dirname.is_file():
        ext = Path(filename_or_dirname).suffix
        ext_list = [ext[1:]]
    else:
        ext_list = list({filename.suffix[1:] for filename in filename_or_dirname.glob("*") if filename.is_file()})

    possibles = []
    for ext in ext_list:
        for rawio in rawiolist:
            if any(ext.lower() == ext2.lower() for ext2 in rawio.extensions):
                possibles.append(rawio)

    if len(possibles) == 1 and exclusive_rawio:
        return possibles[0]
    elif exclusive_rawio:
        return None
    else:
        possibles = [io[0] for io in Counter(possibles).most_common()]
        return possibles
