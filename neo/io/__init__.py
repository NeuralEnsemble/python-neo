# -*- coding: utf-8 -*-
"""
:mod:`neo.io` provides classes for reading and/or writing
electrophysiological data files.

Note that if the package dependency is not satisfied for one io, it does not
raise an error but a warning.

:attr:`neo.io.iolist` provides a list of successfully imported io classes.

Functions:

.. autofunction:: neo.io.get_io


Classes:

.. autoclass:: neo.io.AlphaOmegaIO

.. autoclass:: neo.io.AsciiSignalIO

.. autoclass:: neo.io.AsciiSpikeTrainIO

.. autoclass:: neo.io.AxographIO

.. autoclass:: neo.io.AxonIO

.. autoclass:: neo.io.BCI2000IO

.. autoclass:: neo.io.BlackrockIO

.. autoclass:: neo.io.BrainVisionIO

.. autoclass:: neo.io.BrainwareDamIO

.. autoclass:: neo.io.BrainwareF32IO

.. autoclass:: neo.io.BrainwareSrcIO

.. autoclass:: neo.io.ElanIO

.. autoclass:: neo.io.ElphyIO

.. autoclass:: neo.io.IgorIO

.. autoclass:: neo.io.KlustaKwikIO

.. autoclass:: neo.io.KwikIO

.. autoclass:: neo.io.MicromedIO

.. autoclass:: neo.io.NeoHdf5IO

.. autoclass:: neo.io.NeoMatlabIO

.. autoclass:: neo.io.NestIO

.. autoclass:: neo.io.NeuralynxIO

.. autoclass:: neo.io.NeuroExplorerIO

.. autoclass:: neo.io.NeuroScopeIO

.. autoclass:: neo.io.NeuroshareIO

.. autoclass:: neo.io.NixIO

.. autoclass:: neo.io.NSDFIO

.. autoclass:: neo.io.PickleIO

.. autoclass:: neo.io.PlexonIO

.. autoclass:: neo.io.RawBinarySignalIO

.. autoclass:: neo.io.StimfitIO

.. autoclass:: neo.io.TdtIO

.. autoclass:: neo.io.WinEdrIO

.. autoclass:: neo.io.WinWcpIO

"""

import os.path

# try to import the neuroshare library.
# if it is present, use the neuroshareapiio to load neuroshare files
# if it is not present, use the neurosharectypesio to load files
try:
    import neuroshare as ns
except ImportError as err:
    from neo.io.neurosharectypesio import NeurosharectypesIO as NeuroshareIO
    # print("\n neuroshare library not found, loading data with ctypes" )
    # print("\n to use the API be sure to install the library found at:")
    # print("\n www.http://pythonhosted.org/neuroshare/")

else:
    from neo.io.neuroshareapiio import NeuroshareapiIO as NeuroshareIO
    # print("neuroshare library successfully imported")
    # print("\n loading with API...")

from neo.io.alphaomegaio import AlphaOmegaIO
from neo.io.asciisignalio import AsciiSignalIO
from neo.io.asciispiketrainio import AsciiSpikeTrainIO
from neo.io.axonio import AxonIO
from neo.io.axographio import AxographIO
from neo.io.blackrockio import BlackrockIO
from neo.io.blackrockio_v4 import BlackrockIO as OldBlackrockIO
from neo.io.bci2000io import BCI2000IO
from neo.io.brainvisionio import BrainVisionIO
from neo.io.brainwaredamio import BrainwareDamIO
from neo.io.brainwaref32io import BrainwareF32IO
from neo.io.brainwaresrcio import BrainwareSrcIO
from neo.io.elanio import ElanIO
# from neo.io.elphyio import ElphyIO
from neo.io.exampleio import ExampleIO
from neo.io.igorproio import IgorIO
from neo.io.klustakwikio import KlustaKwikIO
from neo.io.kwikio import KwikIO
from neo.io.micromedio import MicromedIO
from neo.io.hdf5io import NeoHdf5IO
from neo.io.neomatlabio import NeoMatlabIO
from neo.io.nestio import NestIO
from neo.io.neuralynxio import NeuralynxIO
from neo.io.neuralynxio_v1 import NeuralynxIO as OldNeuralynxIO
from neo.io.neuroexplorerio import NeuroExplorerIO
from neo.io.neuroscopeio import NeuroScopeIO
from neo.io.nixio import NixIO
from neo.io.nixio_fr import NixIO as NixIOFr
from neo.io.nsdfio import NSDFIO
from neo.io.pickleio import PickleIO
from neo.io.plexonio import PlexonIO
from neo.io.rawbinarysignalio import RawBinarySignalIO
from neo.io.spike2io import Spike2IO
from neo.io.stimfitio import StimfitIO
from neo.io.tdtio import TdtIO
from neo.io.winedrio import WinEdrIO
from neo.io.winwcpio import WinWcpIO

iolist = [
    AlphaOmegaIO,
    AsciiSignalIO,
    AsciiSpikeTrainIO,
    AxonIO,
    AxographIO,
    BCI2000IO,
    BlackrockIO,
    BrainVisionIO,
    BrainwareDamIO,
    BrainwareF32IO,
    BrainwareSrcIO,
    ElanIO,
    # ElphyIO,
    ExampleIO,
    IgorIO,
    KlustaKwikIO,
    KwikIO,
    MicromedIO,
    NixIO,  # place NixIO before NeoHdf5IO to make it the default for .h5 files
    NeoHdf5IO,
    NeoMatlabIO,
    NestIO,
    NeuralynxIO,
    NeuroExplorerIO,
    NeuroScopeIO,
    NeuroshareIO,
    NSDFIO,
    PickleIO,
    PlexonIO,
    RawBinarySignalIO,
    Spike2IO,
    StimfitIO,
    TdtIO,
    WinEdrIO,
    WinWcpIO
]


def get_io(filename, *args, **kwargs):
    """
    Return a Neo IO instance, guessing the type based on the filename suffix.
    """
    extension = os.path.splitext(filename)[1][1:]
    for io in iolist:
        if extension in io.extensions:
            return io(filename, *args, **kwargs)

    raise IOError("File extension %s not registered" % extension)
