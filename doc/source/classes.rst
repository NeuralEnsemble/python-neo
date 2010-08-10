***********
Neo Classes
***********

.. currentmodule:: neo

Introduction
============

The neo core consist of a collection of classes that define standards object for manipulating electrophysilogical ( in vivo or simulated) data sets.

.. image:: ./base_schematic.png
   :height: 500 px
   :alt: Neo : Neurotools/OpenElectrophy shared base architecture 
   :align: center

This structure attempts to encapsulate the essence of the base data structures previously used
in `Neurotools <http://neuralensemble.org/trac/NeuroTools>`_ and in `OpenElectrophy <http://neuralensemble.org/trac/OpenElectrophy>`_. 
We also attempt to keep a model similar to the one described by the `Neuroshare <http://neuroshare.sourceforge.net/index.shtml standard>`_ IO API.


Features
====================

You only have to known this list all the rest is intuitive:
 * As neo is more a nomenclature than a complete tools suite for analysing. In a first time, neo objects will stay simple, for complicate behavior you need to subclass neo objects.
 * Some ojects are kind of container : Block, Segment, Neuron, SpikeTrainList, AnalogSignalList, ...
 * And some oject hold data and have some standart behavior : AnalogSignal, SpikeTrain, Spike ...
 * For all objects, some attributes are necessary and some other are optionals.
 Example: AnalogSignal:
  * Neccessary  : signal, sampling_rate, t_start
  * Optional : channel, name, ...
 * A container can acces its descendance with get_xxxxxxxx() or with the property _xxxxxxxxxxxx (where xxx is the child name)
 Example :
  * Block.get_segments() or Block._segments
  * Segment.getanalogsignals() or Segment._analosignals
  * Neuron.get_spiketrains() or Neuron._spiketrains
  
 
 


Detailed description of neo classes
===================================

The best to understand neo is definition a each class:

.. autoclass:: neo.Block
.. autoclass:: neo.Segment
.. autoclass:: neo.Neuron
.. autoclass:: neo.Event
.. autoclass:: neo.Epoch
.. autoclass:: neo.SpikeTrain
.. autoclass:: neo.SpikeTrainList
.. autoclass:: neo.AnalogSignal
.. autoclass:: neo.AnalogSignalList
.. autoclass:: neo.RecordingPoint






