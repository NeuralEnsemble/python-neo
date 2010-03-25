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


from block import Block
from segment import Segment
from event import Event
from epoch import Epoch
from neuron import Neuron
from spiketrain import SpikeTrain
from spiketrainlist import SpikeTrainList
from analogsignal import AnalogSignal
from analogsignallist import AnalogSignalList
from spike import Spike
from recordingpoint import RecordingPoint


neotypes = [ Block , Segment , AnalogSignal, Event, Epoch, Neuron, SpikeTrain, SpikeTrainList , Spike , RecordingPoint ]
