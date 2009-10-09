"""
Neo.core
==================

A collection of functions to create, manipulate and play with analog signals. 

Classes
-------

AnalogSignal     - object representing an analog signal, with its data. Can be used to do 
                   threshold detection, event triggered averages, ...
AnalogSignalList - list of AnalogSignal objects, again with methods such as mean, std, plot, 
                   and so on
Epoch
Block
Neuron
Segment
Spike
SpikeTrain
SpikeTrainList
Event
RecordingPoint
"""


from block import *
from event import *
from epoch import *
from neuron import *
from spiketrain import *
from spiketrainlist import *
from analogsignal import *
from analogsignallist import *
from spike import *
from recordingpoint import *
