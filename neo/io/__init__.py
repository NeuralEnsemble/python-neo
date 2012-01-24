# encoding: utf-8
"""
neo.io provides classes for reading and/or writing electrophysiological data files.

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


"""
# AND also  .. autoclass:: neo.io.NeoHdf5IO


import warnings

iolist = []

try:
    from .exampleio import ExampleIO
    iolist.append( ExampleIO )
except ImportError:
    warnings.warn("ExampleIO not available, check dependencies", ImportWarning)

try:
    from .hdf5io import NeoHdf5IO
    iolist.append( NeoHdf5IO )
except ImportError:
    warnings.warn("NeoHdf5IO not available, check dependencies", ImportWarning)

try:
    from .plexonio import PlexonIO
    iolist.append( PlexonIO )
except ImportError:
    warnings.warn("PlexonIO not available, check dependencies", ImportWarning)

try:
    from .neuroexplorerio import NeuroExplorerIO
    iolist.append( NeuroExplorerIO )
except ImportError:
    warnings.warn("NeuroExplorerIO not available, check dependencies", ImportWarning)

try:
    from .axonio import AxonIO
    iolist.append( AxonIO )
except ImportError:
    warnings.warn("AxonIO not available, check dependencies", ImportWarning)

try:
    from .tdtio import TdtIO
    iolist.append( TdtIO )
except ImportError:
    warnings.warn("TdtIO not available, check dependencies", ImportWarning)

try:
    from .spike2io import Spike2IO
    iolist.append( Spike2IO )
except ImportError:
    warnings.warn("Spike2IO not available, check dependencies", ImportWarning)

try:
    from .winedrio import WinEdrIO
    iolist.append( WinEdrIO )
except ImportError:
    warnings.warn("WinEdrIO not available, check dependencies", ImportWarning)

try:
    from .winwcpio import WinWcpIO
    iolist.append( WinWcpIO )
except ImportError:
    warnings.warn("WinWcpIO not available, check dependencies", ImportWarning)

try:
    from .elanio import ElanIO
    iolist.append( ElanIO )
except ImportError:
    warnings.warn("ElanIO not available, check dependencies", ImportWarning)

try:
    from .asciisignalio import AsciiSignalIO
    iolist.append( AsciiSignalIO )
except ImportError:
    warnings.warn("AsciiSignalIO not available, check dependencies", ImportWarning)

try:
    from .asciispiketrainio import AsciiSpikeTrainIO
    iolist.append( AsciiSpikeTrainIO )
except ImportError:
    warnings.warn("AsciiSpikeTrainIO not available, check dependencies", ImportWarning)

try:
    from .rawbinarysignalio import RawBinarySignalIO
    iolist.append( RawBinarySignalIO )
except ImportError:
    warnings.warn("RawBinarySignalIO not available, check dependencies", ImportWarning)

try:
    from .micromedio import MicromedIO
    iolist.append( MicromedIO )
except ImportError:
    warnings.warn("MicromedIO not available, check dependencies", ImportWarning)

try:
    from .neuroshareio import NeuroshareIO
    iolist.append( NeuroshareIO )
except ImportError:
    warnings.warn("NeuroshareIO not available, check dependencies", ImportWarning)

try:
    from .neomatlabio import NeoMatlabIO
    iolist.append( NeoMatlabIO )
except ImportError:
    warnings.warn("NeoMatlabIO not available, check dependencies", ImportWarning)

try:
    from .pynnio import PyNNNumpyIO
    iolist.append( PyNNNumpyIO )
except ImportError:
    warnings.warn("PyNNNumpyIO not available, check dependencies", ImportWarning)

try:
    from .pynnio import PyNNTextIO
    iolist.append( PyNNTextIO )
except ImportError:
    warnings.warn("PyNNTextIO not available, check dependencies", ImportWarning)

try:
    from .klustakwikio import KlustaKwikIO
    iolist.append( KlustaKwikIO )
except ImportError:
    warnings.warn("KlustaKwikIO not available, check dependencies", ImportWarning)

try:
    from .blackrockio import BlackrockIO
    iolist.append( BlackrockIO )
except ImportError:
    warnings.warn("BlackrockIO not available, check dependencies", ImportWarning)

try:
    from .alphaomegaio import AlphaOmegaIO
    iolist.append( AlphaOmegaIO )
except ImportError:
    warnings.warn("AlphaOmegaIO not available, check dependencies", ImportWarning)


