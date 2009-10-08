# -*- coding: utf-8 -*-
"""
Neo
==================

A collection of functions to create, manipulate and play with spikes signals. 

Classes
-------

SpikeTrain       - object representing a spike train, for one cell. Useful for plots, 
                   calculations such as ISI, CV, mean rate(), ...
SpikeList        - object representing the activity of a population of neurons. Functions as a
                   dictionary of SpikeTrain objects, with methods to compute firing rate,
                   ISI, CV, cross-correlations, and so on.

Functions
---------

load_spikelist       - load a SpikeList object from a file. Expects a particular format.
                       Can also load data in a different format, but then you have
                       to write your own File object that will know how to read the data (see io.py)
load                 - a generic loader for all the previous load methods.

See also NeuroTools.signals.analogs
"""

from neuron import Neuron
