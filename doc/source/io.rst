***********
Neo IO
***********

.. currentmodule:: neo

Introduction
===============

The Neo IO module aims to provide an exhaustive way of loading and saving several widely used data format in Electrophysiology. The more those heterogeneous formats will be supported, the more easy it will be to manipulate them as Neo objects in a similar way. Therefore the IO set of classes propose a simple and flexible IO API that fit many formats specifications. It is not only file oriented, it can also read/write objects from database.

Detail of API
================

The neo.io API is design to be simple and intuitive:
 - each file format has its IO classes (for examples for Spike2 files you have a Spike2IO class)
 - each class inherit BaseIO class
 - each io class can read or write one or several neo objects (for example Segment, SpikeTrainList, Block, ...)
 - each io class should therefore declare what type of objects it is able to read or write.


.. autoclass:: neo.io.baseio.BaseIO


Examples of use
================

The basic syntax is as follow. If you want to load a file format which is implemented in a MyFormatIO class


>> from neo.io.myformat import MyFormatIO

>> file = MyFormatIO("myfile.dat")

To know what types of objects are supported by this io interface:

>> file.supported_objects

And then for example to read a segment in this file format (if :class:`Segment` is supported)

>> file.read_segment()


List of implemented formats
=================================

 - :class:`AsciiSignalIO`
    .. automodule:: neo.io.asciisignalio

 - :class:`AsciiSpikeIO`
    .. automodule:: neo.io.asciispikeio

 - :class:`AxonIO`
    .. automodule:: neo.io.axonio

 - :class:`EegLabIO`
    .. automodule:: neo.io.eeglabio

 - :class:`ElanIO`
    .. automodule:: neo.io.elanio

 - :class:`ExampleIO`
    .. automodule:: neo.io.exampleio

 - :class:`MicromedIO`
    .. automodule:: neo.io.micromedio

 - :class:`RawIO`
    .. automodule:: neo.io.rawio

 - :class:`Spike2IO`
    .. automodule:: neo.io.spike2io

 - :class:`WinWcpIO`
    .. automodule:: neo.io.winwcpio
    
 - :class:`NeuroshareSpike2IO`
 - :class:`NeurosharePlexonIO`
 - :class:`NeuroshareAlphaOmegaIO`
    .. automodule:: neo.io.neuroshare.neuroshareio
 
 - :class:`PyNNIO`
    .. automodule:: neo.io.pynnio
 
 - :class:`PyNNBinaryIO`
    .. automodule:: neo.io.pynnbinaryio
 
Impletation of a new IO
===========================

ExampleIO is a fake IO just for illustrating how to implement a IO.

.. automodule:: neo.io.exampleyio

.. autoclass:: neo.io.ExampleIO

