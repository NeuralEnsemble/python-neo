# -*- coding: utf-8 -*-
"""
:mod:`neo.io` provides classes for reading and/or writing
electrophysiological data files.

Note that if the package dependency is not satisfied for one io, it does not
raise an error but a warning.

neo.io.iolist provides a list of succesfully imported io classes.

Classes:

.. autoclass:: neo.io.AlphaOmegaIO

.. autoclass:: neo.io.AsciiSignalIO

.. autoclass:: neo.io.AsciiSpikeTrainIO

.. autoclass:: neo.io.AxonIO

.. autoclass:: neo.io.BlackrockIO

.. autoclass:: neo.io.BrainVisionIO

.. autoclass:: neo.io.BrainwareDamIO

.. autoclass:: neo.io.BrainwareF32IO

.. autoclass:: neo.io.BrainwareSrcIO

.. autoclass:: neo.io.ElanIO

.. autoclass:: neo.io.ElphyIO

.. autoclass:: neo.io.KlustaKwikIO

.. autoclass:: neo.io.MicromedIO

.. autoclass:: neo.io.NeoHdf5IO

.. autoclass:: neo.io.NeoMatlabIO

.. autoclass:: neo.io.NeuroExplorerIO

.. autoclass:: neo.io.NeuroScopeIO

.. autoclass:: neo.io.NeuroshareIO

.. autoclass:: neo.io.PickleIO

.. autoclass:: neo.io.PlexonIO

.. autoclass:: neo.io.PyNNNumpyIO

.. autoclass:: neo.io.PyNNTextIO

.. autoclass:: neo.io.RawBinarySignalIO

.. autoclass:: neo.io.TdtIO

.. autoclass:: neo.io.WinEdrIO

.. autoclass:: neo.io.WinWcpIO

"""

import os.path


from neo.io.alphaomegaio import AlphaOmegaIO
from neo.io.asciisignalio import AsciiSignalIO
from neo.io.asciispiketrainio import AsciiSpikeTrainIO
from neo.io.axonio import AxonIO
from neo.io.blackrockio import BlackrockIO
from neo.io.brainvisionio import BrainVisionIO
from neo.io.brainwaredamio import BrainwareDamIO
from neo.io.brainwaref32io import BrainwareF32IO
from neo.io.brainwaresrcio import BrainwareSrcIO
from neo.io.elanio import ElanIO
from neo.io.elphyio import ElphyIO
from neo.io.exampleio import ExampleIO
from neo.io.klustakwikio import KlustaKwikIO
from neo.io.micromedio import MicromedIO
from neo.io.hdf5io import NeoHdf5IO
from neo.io.neomatlabio import NeoMatlabIO
from neo.io.neuroexplorerio import NeuroExplorerIO
from neo.io.neuroscopeio import NeuroScopeIO
from neo.io.neuroshareio import NeuroshareIO
from neo.io.pickleio import PickleIO
from neo.io.plexonio import PlexonIO
from neo.io.pynnio import PyNNNumpyIO
from neo.io.pynnio import PyNNTextIO
from neo.io.rawbinarysignalio import RawBinarySignalIO
from neo.io.spike2io import Spike2IO
from neo.io.tdtio import TdtIO
from neo.io.winedrio import WinEdrIO
from neo.io.winwcpio import WinWcpIO


iolist = [AlphaOmegaIO,
          AsciiSignalIO,
          AsciiSpikeTrainIO,
          AxonIO,
          BlackrockIO,
          BrainVisionIO,
          BrainwareDamIO,
          BrainwareF32IO,
          BrainwareSrcIO,
          ElanIO,
          ElphyIO,
          ExampleIO,
          KlustaKwikIO,
          MicromedIO,
          NeoHdf5IO,
          NeoMatlabIO,
          NeuroExplorerIO,
          NeuroScopeIO,
          NeuroshareIO,
          PickleIO,
          PlexonIO,
          PyNNNumpyIO,
          PyNNTextIO,
          RawBinarySignalIO,
          Spike2IO,
          TdtIO,
          WinEdrIO,
          WinWcpIO]


def get_io(filename):
    """
    Return a Neo IO instance, guessing the type based on the filename suffix.
    """
    extension = os.path.splitext(filename)[1][1:]
    for io in iolist:
        if extension in io.extensions:
            return io(filename=filename)

    raise IOError("file extension %s not registered" % extension)
