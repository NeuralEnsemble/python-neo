# -*- coding: utf-8 -*-

"""
neo.io
==================

A collection of classes for reading/writing as many as possible formats for
electrophysiological dataset.

Classes
-------

BaseFile        - abstract class which should be overriden, managing how a file will load/write
                  its data
"""

from baseio import *

all_format = [ ]


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
    from elphyio import ElphyIO
    all_format += [ [ 'elphy DAT' , { 'class' : ElphyIO  , 'info' :  'DAT elphy file' } ] ]
except ImportError:
    print "Error while loading ElphyIO module"

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
    



#~ try:
    #~ from pynn import TextFile
    #~ all_format += ['PyNN']
#~ except ImportError:
    #~ print "Error while loading pyNN IO module"

#try:
    #from spike2 import *
    #all_format += ['PyNN']
#except ImportError:
    #pass



#~ if sys.platform =='win32':
	#~ from neurshare import Neuroshare
	#~ all_IOclass += [ Neuroshare ]
