***********
Neo IO
***********

.. currentmodule:: neo

Introduction
===============

The Neo IO module aims to provide an exhaustive way of loading and saving several widely used data formats in electrophysiology.
The more these heterogeneous formats are supported, the easier it will be to manipulate them as Neo objects in a similar way.
Therefore the IO set of classes propose a simple and flexible IO API that fits many format specifications.
It is not only file oriented, it can also read/write objects from a database.

Details of API
================

The neo.io API is designed to be simple and intuitive:
 - each file format has its IO classes (for example for Spike2 files you have a Spike2IO class)
 - each class inherits from the BaseIO class
 - each io class can read or write directly one or several neo objects (for example Segment, SpikeTrainList, Block, ...)
 - each io class support part of the neo.core hierachy not necessary all part.
 - each io is able to to a *easy* load and or a *cascade* load

There is an intrinsic structure in the different Neo objects, that could be seen as a hierachy.
The highest level object is the :class:`Block` object, which is the high level container able to encapsulate all the others.

A :class:`Block` is therefore a list of :class:`Segment` objects, that can, in some file formats, be accessed individually.
Depending on the file format, i.e. if it is streamable or not, the whole :class:`Block` may need to be loaded, but sometimes 
particular :class:`Segment` objects can be accessed individually.
Within a :class:`Segment`, the same hierarchical organisation applies.
A :class:`Segment` embeds several objects, such as :class:`SpikeTrain`  :class:`AnalogSignal`, :class:`AnaloSignalList`, :class:`Epoch`, :class:`Event` (basically, all the different Neo objects).

Depending on the file format, these objects can sometimes be loaded separately, without the need to load the whole file.
If possible, a file IO therefore provides distinct methods allowing to load only particular objects that may be present in the file.
The basic idea of each IO file format is to have as much as possible read/write methods for the encapsulated individual objects, 
and otherwise, to provide a read/write method that will return the object at the highest level of hierarchy (by default, a :class:`Block` or a :class:`Segment`).

.. autoclass:: neo.io.baseio.BaseIO


Examples of use
================

The basic syntax is as follows. If you want to load a file format that is implemented in a generic MyFormatIO class::

    >>> from neo.io import MyFormatIO
    >>> reader = MyFormatIO("myfile.dat")

To know what types of object are supported by this io interface::

    >>> file.supported_objects
    [Segment , AnalogSignal , SpikeTrain, Event, Spike ]

Supported objects does not mean objects that you can read directly. For instance, many formats support AnalogSignal
but you can't access them directly. To access your AnalogSignal, you must read a Segment with the **cascade** set to
True::

    >>> seg = reader.read_segment(cascade=True)
    >>> print seg._analogsignals

To get a list of directly readable objects ::

    >>> reader.readable_objects
    [Segment]

The first element of the previous list is the highest level for reading the file.

To read the whole file::

    >>> result = reader.read()
    >>> type(result)
    neo.core.Segment

In this case, this is equivalent to ::

    >>> seg = reader.read_segment()


In some case you may not want to load everything in memory because it could be to big, in this case you could use the **lazy** flag set
to True or Fasle::
    
    >>>> seg = reader.read_segment(lazy = Fasle, cascade=True)

In later case segment and subhierachy is read but analogsignal are empty.



List of implemented formats
=================================


    


Implementation of a new IO
===========================

ExampleIO is a fake IO just for illustrating how to implement a IO. Developers who would like to write their own file format
should take a look to better understand the structure. One can also refer to the :class:`BaseIO` generic file.

As was already said, the default read/write methods should return the highest object in the hierarchy, 
which is often a :class:`Block` or a :class:`Segment`. Individual read/write methods should be implemented as much as possible.

.. autoclass:: neo.io.ExampleIO

For advice or comments on the coding guidelines, developers can send e-mail to sgarcia@olfac.univ-lyon1.fr, yger@unic.cnrs-gif.fr or estebanez@unic.cnrs-gif.fr.
