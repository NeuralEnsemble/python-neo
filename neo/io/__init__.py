# -*- coding: utf-8 -*-

"""
neo.io
==================

A collection of classes for reading/writing as many as possible formats for
electrophysiological dataset.

neo.io.all_format contain a list of all IO classes.



Classes
-------

"""

#from baseio import *
import sys
all_format = [ ]


# all IO import are inside try .. except to prevent module bugs because of one IO.

try:
    from tryitio import TryItIO
    all_format += [ [ 'tryit' , { 'class' : TryItIO  , 'info' :  'a fake file reader for trying OpenElectrophy' } ] ]
except ImportError:
    print "Error while loading ExampleIO module"


try:
    from exampleio import ExampleIO
    all_format += [ [ 'example' , { 'class' : ExampleIO  , 'info' :  'a fake file reader for example' } ] ]
except ImportError:
    print "Error while loading ExampleIO module"




try:
    from rawio import RawIO
    all_format += [ [ 'raw binary' , { 'class' : RawIO  , 'info' :  'Compact raw binary generic file' } ] ]
except ImportError:
    print "Error while loading RawIO module"


try:
    from asciisignalio import AsciiSignalIO
    all_format += [ [ 'ascii signal' , { 'class' : AsciiSignalIO  , 'info' :  'Ascii Signal generic file' } ] ]
except ImportError:
    print "Error while loading AsciiSignalIO module"

try:
    from asciispikeio import AsciiSpikeIO
    all_format += [ [ 'ascii spike' , { 'class' : AsciiSpikeIO  , 'info' :  'Ascii spike file' } ] ]
except ImportError:
    print "Error while loading AsciiSpikeIO module"

try:
    from micromedio import MicromedIO
    all_format += [ [ 'micromed' , { 'class' : MicromedIO  , 'info' :  'TRC micromed file' } ] ]
except ImportError:
    print "Error while loading MicromedIO module"

try:
    from elphydatio import ElphyDatIO
    all_format += [ [ 'elphy DAT' , { 'class' : ElphyDatIO  , 'info' :  'DAT elphy file' } ] ]
except ImportError:
    print "Error while loading ElphyDatIO module"

try:
    from elanio import ElanIO
    all_format += [ [ 'elan eeg' , { 'class' : ElanIO  , 'info' :  'eeg elan file' } ] ]
except ImportError:
    print "Error while loading ElanIO module"

try:
    from eeglabio import EegLabIO
    all_format += [ [ 'eeglab matlab' , { 'class' : EegLabIO  , 'info' :  'eeglab matlab file' } ] ]
except ImportError:
    print "Error while loading EegLabIO module"

try:
    from axonio import AxonIO
    all_format += [ [ 'axon abf' , { 'class' : AxonIO  , 'info' :  'axon binary file (abf)' } ] ]
except ImportError:
    print "Error while loading AxonIO module"

try :
    from spike2io import Spike2IO
    all_format += [ [ 'Spike2 smr' , { 'class' : Spike2IO  , 'info' :  'CED spike2 file (smr)' } ] ]
except ImportError:
    print "Error while loading Spike2IO module"

try :
    from winwcpio import WinWcpIO
    all_format += [ [ 'WinWcp' , { 'class' : WinWcpIO  , 'info' :  'WinWcp file (wcp)' } ] ]
except ImportError:
    print "Error while loading WinWcpIO module"

try :
    from nexio import NexIO
    all_format += [ [ 'NexIO' , { 'class' : NexIO  , 'info' :  'NeuroExplorer file (nex)' } ] ]
except ImportError:
    print "Error while loading NexIO module"


try :
    from plexonio import PlexonIO
    all_format += [ [ 'PlexonIO' , { 'class' : PlexonIO  , 'info' :  'Plexon file (plx)' } ] ]
except ImportError:
    print "Error while loading PlexonIO module"


try :
    from pynnio import PyNNIO
    all_format += [ [ 'PyNN Text' , { 'class' : PyNNIO  , 'info' :  'PyNN Text file (pynn)' } ] ]
except ImportError:
    print "Error while loading PyNNIO module"

try :
    from pynnbinaryio import PyNNBinaryIO
    all_format += [ [ 'PyNN Numpy Binary' , { 'class' : PyNNBinaryIO  , 'info' :  'PyNN Numpy Binary file (pynn)' } ] ]
except ImportError:
    print "Error while loading PyNNBinaryIO module"


# Specific platform IO : neuroshare DLLs

if sys.platform =='win32':
    try :
        from neuroshare.neuroshareio import NeuroshareSpike2IO
        all_format += [ [ 'Spike2 smr' , { 'class' : NeuroshareSpike2IO  , 'info' :  'CED spike2 file (smr) neuroshare' } ] ]
    except ImportError:
        print "Error while loading NeuroshareSpike2IO module"
    

    try :
        from neuroshare.neuroshareio import NeurosharePlexonIO
        all_format += [ [ 'Spike2 smr' , { 'class' : NeurosharePlexonIO  , 'info' :  'plexon file (nex) neuroshare' } ] ]
    except ImportError:
        print "Error while loading NeurosharePlexonIO module"

    try :
        from neuroshare.neuroshareio import NeuroshareAlphaOmegaIO
        all_format += [ [ 'Spike2 smr' , { 'class' : NeuroshareAlphaOmegaIO  , 'info' :  'AlphaOmega file (map) neuroshare' } ] ]
    except ImportError:
        print "Error while loading NeuroshareAlphaOmegaIO module"

    try :
        from neuroshare.neuroshareio import NeuroshareTdtIO
        all_format += [ [ 'Spike2 smr' , { 'class' : NeuroshareTdtIO  , 'info' :  'TDT tank  neuroshare' } ] ]
    except ImportError:
        print "Error while loading NeuroshareTdtIO module"



	
