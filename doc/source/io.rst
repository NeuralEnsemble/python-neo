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

There is an intrinsic structure in the different Neo objects, that could be seen as a hierachy. The highest level object is the :class:`Block` object, which is the high level container able to encapsulate all the others.

A :class:`Block` is therefore a list of :class:`Segment` objects, that can, in some file format, be accessed individually. Depending on the file format, if it is streamable or not, the whole :class:`Block` need to be loaded, but sometimes, only
a particular :class:`Segment` can be accessed. Within a :class:`Segment`, the same hierarchical organisation applies. A :class:`Segment` embbed several objects, such as :class:`SpikeTrain` :class:`SpikeTrainList`, :class:`AnalogSignal`, :class:`AnaloSignalList`, :class:`Epoch`, :class:`Event` (basically, all the differents Neo objects).

Depending on the file format, those objects can sometimes be loaded separately, without the need to load the whole file. If possible, a file IO provides therefore distincts methods allowing to load only particular objects that may be present in the file. The basic idea of each IO file format is to have as much as possible read/write methods for those encapsulated individual objecs, and otherwise, to provide a read/write method that will return the object at the highest level of hierarchy. By default, it should be a :class:`Block` or a :class:`Segment`.

.. autoclass:: neo.io.baseio.BaseIO


Examples of use
================

The basic syntax is as follow. If you want to load a file format which is implemented in a MyFormatIO class


>>> from neo.io import MyFormatIO
>>> file = MyFormatIO("myfile.dat")

To know what types of objects are supported by this io interface:

>>> file.supported_objects
[Segment , AnalogSignal , SpikeTrain, Event, Spike ]

Supported objects, do not meens objects that you can read directly. For instance, many formats supports AnalogSignal
but you can't acces them directly : you must read a Segment and acces your AnalogSignal like that :

>>> seg = file.read_segment()
>>> seg.get_analogsignals()

To have the list of directly readeable objects :

>>> file.readable_objects
[Segment]

The first element of the previous list is the highest level for reading the file.

To read all the file :

>>> result = file.read()
>>> type(result)
neo.core.Segment

In this case, this is equivalent to :
>>> seg = file.read_segment()



List of implemented formats
=================================

 - :class:`PlexonIO`
    .. automodule:: neo.io.plexonio
    
 - :class:`Spike2IO`
    .. automodule:: neo.io.spike2io

 - :class:`NexIO`
    .. automodule:: neo.io.nexio

 - :class:`AxonIO`
    .. automodule:: neo.io.axonio

 - :class:`MicromedIO`
    .. automodule:: neo.io.micromedio

 - :class:`AsciiSignalIO`
    .. automodule:: neo.io.asciisignalio

 - :class:`NeuroshareSpike2IO`
 - :class:`NeurosharePlexonIO`
 - :class:`NeuroshareAlphaOmegaIO`
    .. automodule:: neo.io.neuroshare.neuroshareio

 - :class:`EegLabIO`
    .. automodule:: neo.io.eeglabio


 - :class:`ExampleIO`
    .. automodule:: neo.io.exampleio

 - :class:`RawIO`
    .. automodule:: neo.io.rawio

 - :class:`WinWcpIO`
    .. automodule:: neo.io.winwcpio

 - :class:`ElanIO`
    .. automodule:: neo.io.elanio

 
 - :class:`PyNNIO`
    .. automodule:: neo.io.pynnio
 
 - :class:`PyNNBinaryIO`
    .. automodule:: neo.io.pynnbinaryio

 - :class:`AsciiSpikeIO`
    .. automodule:: neo.io.asciispikeio

    


Impletation of a new IO
===========================

ExampleIO is a fake IO just for illustrating how to implement a IO. For developpers that would like to write their own file format,
they should have a look to better catch the structure. One can also refer to the :class:`BaseIO` generic file.

As already said, the default read/write methods should return the highest object in the hierarchy, 
which is often a :class:`Block` or a :class:`Segment`. Individuals read/write methods should be implemented as much as possible.

.. autoclass:: neo.io.ExampleIO

For any advices or hints on the coding guidelines, developpers can send mails to sgarcia@olfac.univ-lyon1.fr, yger@unic.cnrs-gif.fr, estebanez@unic.cnrs-gif.fr
