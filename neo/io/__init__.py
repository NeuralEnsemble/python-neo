# encoding: utf-8
"""
neo.io provide classes for reading and/or writing electrophysiological data files.

It is a pure python neuroshare remplacement.

Note that if the package dependency is not satisfiyed for one io, it do not raise
a error but a warning. In short dependecy for io are recommendation and not obligation.


neo.io.iolist provide the classes list of succecfull imported io.

"""

import warnings

iolist = [ ]




try:
    from exampleio import ExampleIO
    iolist.append( ExampleIO )
except ImportError:
    warnings.warn("ExampleIO not available, check dependencies", ImportWarning)


try:
    from plexonio import PlexonIO
    iolist.append( PlexonIO )
except ImportError:
    warnings.warn("PlexonIO not available, check dependencies", ImportWarning)


try:
    from neuroexplorerio import NeuroExplorerIO
    iolist.append( NeuroExplorerIO )
except ImportError:
    warnings.warn("NeuroExplorerIO not available, check dependencies", ImportWarning)


try:
    from axonio import AxonIO
    iolist.append( AxonIO )
except ImportError:
    warnings.warn("AxonIO not available, check dependencies", ImportWarning)

try:
    from tdtio import TdtIO
    iolist.append( TdtIO )
except ImportError:
    warnings.warn("TdtIO not available, check dependencies", ImportWarning)

try:
    from spike2io import Spike2IO
    iolist.append( Spike2IO )
except ImportError:
    warnings.warn("Spike2IO not available, check dependencies", ImportWarning)

try:
    from winedrio import WinEdrIO
    iolist.append( WinEdrIO )
except ImportError:
    warnings.warn("WinEdrIO not available, check dependencies", ImportWarning)

try:
    from winwcpio import WinWcpIO
    iolist.append( WinWcpIO )
except ImportError:
    warnings.warn("WinWcpIO not available, check dependencies", ImportWarning)

try:
    from elanio import ElanIO
    iolist.append( ElanIO )
except ImportError:
    warnings.warn("ElanIO not available, check dependencies", ImportWarning)


