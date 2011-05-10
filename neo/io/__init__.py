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
    warnings.warn("ExampleIO not  available", ImportWarning)


try:
    from plexonio import PlexonIO
    iolist.append( PlexonIO )
except ImportError:
    warnings.warn("PlexonIO not  available", ImportWarning)


try:
    from neuroexplorerio import NeuroExplorerIO
    iolist.append( NeuroExplorerIO )
except ImportError:
    warnings.warn("NeuroExplorerIO not  available", ImportWarning)


try:
    from axonio import AxonIO
    iolist.append( AxonIO )
except ImportError:
    warnings.warn("AxonIO not  available", ImportWarning)

try:
    from tdtio import TdtIO
    iolist.append( TdtIO )
except ImportError:
    warnings.warn("TdtIO not  available", ImportWarning)

