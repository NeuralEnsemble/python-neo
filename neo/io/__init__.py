# -*- coding: utf-8 -*-
"""
neo.io provides classes for reading and/or writing
electrophysiological data files.

Note that if the package dependency is not satisfied for one io, it does not
raise an error but a warning.

neo.io.iolist provides the classes list of succesfully imported io.

.. autoclass:: neo.io.PlexonIO

.. autoclass:: neo.io.NeuroExplorerIO

.. autoclass:: neo.io.AxonIO

.. autoclass:: neo.io.TdtIO

.. autoclass:: neo.io.WinEdrIO

.. autoclass:: neo.io.WinWcpIO

.. autoclass:: neo.io.ElanIO

.. autoclass:: neo.io.AsciiSignalIO

.. autoclass:: neo.io.AsciiSpikeTrainIO

.. autoclass:: neo.io.RawBinarySignalIO

.. autoclass:: neo.io.MicromedIO

.. autoclass:: neo.io.NeuroshareIO

.. autoclass:: neo.io.NeoMatlabIO

.. autoclass:: neo.io.PyNNNumpyIO

.. autoclass:: neo.io.PyNNTextIO

.. autoclass:: neo.io.KlustaKwikIO

.. autoclass:: neo.io.BlackrockIO

.. autoclass:: neo.io.AlphaOmegaIO

.. autoclass:: neo.io.PickleIO

.. autoclass:: neo.io.NeoHdf5IO

.. autoclass:: neo.io.BrainVisionIO

.. autoclass:: neo.io.ElphyIO

.. autoclass:: neo.io.NeuroScopeIO

.. autoclass:: neo.io.BrainwareDamIO

.. autoclass:: neo.io.BrainwareF32IO

.. autoclass:: neo.io.BrainwareSrcIO
"""

import warnings
import os.path

iolist = []


try:
    from neo.io.exampleio import ExampleIO
except ImportError:
    warnings.warn("ExampleIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(ExampleIO)


try:
    from neo.io.hdf5io import NeoHdf5IO
except ImportError:
    warnings.warn("NeoHdf5IO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(NeoHdf5IO)


try:
    from neo.io.plexonio import PlexonIO
except ImportError:
    warnings.warn("PlexonIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(PlexonIO)


try:
    from neo.io.neuroexplorerio import NeuroExplorerIO
except ImportError:
    warnings.warn("NeuroExplorerIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(NeuroExplorerIO)


try:
    from neo.io.axonio import AxonIO
except ImportError:
    warnings.warn("AxonIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(AxonIO)


try:
    from neo.io.tdtio import TdtIO
except ImportError:
    warnings.warn("TdtIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(TdtIO)


try:
    from neo.io.spike2io import Spike2IO
except ImportError:
    warnings.warn("Spike2IO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(Spike2IO)


try:
    from neo.io.winedrio import WinEdrIO
except ImportError:
    warnings.warn("WinEdrIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(WinEdrIO)


try:
    from neo.io.winwcpio import WinWcpIO
except ImportError:
    warnings.warn("WinWcpIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(WinWcpIO)


try:
    from neo.io.elanio import ElanIO
except ImportError:
    warnings.warn("ElanIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(ElanIO)


try:
    from neo.io.asciisignalio import AsciiSignalIO
except ImportError:
    warnings.warn("AsciiSignalIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(AsciiSignalIO)


try:
    from neo.io.asciispiketrainio import AsciiSpikeTrainIO
except ImportError:
    warnings.warn("AsciiSpikeTrainIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(AsciiSpikeTrainIO)


try:
    from neo.io.rawbinarysignalio import RawBinarySignalIO
except ImportError:
    warnings.warn("RawBinarySignalIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(RawBinarySignalIO)


try:
    from neo.io.micromedio import MicromedIO
except ImportError:
    warnings.warn("MicromedIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(MicromedIO)


try:
    from neo.io.neuroshareio import NeuroshareIO
except ImportError:
    warnings.warn("NeuroshareIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(NeuroshareIO)


try:
    from neo.io.neomatlabio import NeoMatlabIO
except ImportError:
    warnings.warn("NeoMatlabIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(NeoMatlabIO)


try:
    from neo.io.pynnio import PyNNNumpyIO
except ImportError:
    warnings.warn("PyNNNumpyIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(PyNNNumpyIO)


try:
    from neo.io.pynnio import PyNNTextIO
except ImportError:
    warnings.warn("PyNNTextIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(PyNNTextIO)


try:
    from neo.io.klustakwikio import KlustaKwikIO
except ImportError:
    warnings.warn("KlustaKwikIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(KlustaKwikIO)


try:
    from neo.io.blackrockio import BlackrockIO
except ImportError:
    warnings.warn("BlackrockIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(BlackrockIO)


try:
    from neo.io.alphaomegaio import AlphaOmegaIO
except ImportError:
    warnings.warn("AlphaOmegaIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(AlphaOmegaIO)


# should be always available, so no need for try...except
from neo.io.pickleio import PickleIO
iolist.append(PickleIO)


try:
    from neo.io.brainvisionio import BrainVisionIO
except ImportError:
    warnings.warn("BrainVisionIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(BrainVisionIO)


try:
    from neo.io.elphyio import ElphyIO
except ImportError:
    warnings.warn("ElphyIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(ElphyIO)


try:
    from neo.io.neuroscopeio import NeuroScopeIO
except ImportError:
    warnings.warn("NeuroScopeIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(NeuroScopeIO)


try:
    from neo.io.brainwaresrcio import BrainwareSrcIO
except ImportError:
    warnings.warn("BrainwareSrcIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(BrainwareSrcIO)


try:
    from neo.io.brainwaredamio import BrainwareDamIO
except ImportError:
    warnings.warn("BrainwareDamIO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(BrainwareDamIO)


try:
    from neo.io.brainwaref32io import BrainwareF32IO
except ImportError:
    warnings.warn("BrainwareF32IO not available, check dependencies",
                  ImportWarning)
else:
    iolist.append(BrainwareF32IO)


def get_io(filename):
    """
    Return a Neo IO instance, guessing the type based on the filename suffix.
    """
    extension = os.path.splitext(filename)[1][1:]
    for io in iolist:
        if extension in io.extensions:
            return io(filename=filename)

    raise IOError("file extension %s not registered" % extension)
