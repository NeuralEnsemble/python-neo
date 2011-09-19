***********
Neo IO
***********

.. currentmodule:: neo

Preampbule
===============


The Neo IO module aims to provide an exhaustive way of loading and saving several widely used data formats in electrophysiology.
The more these heterogeneous formats are supported, the easier it will be to manipulate them as Neo objects in a similar way.
Therefore the IO set of classes propose a simple and flexible IO API that fits many format specifications.
It is not only file oriented, it can also read/write objects from a database.

neo.io can be seen as a *pure Python coded* and open source Neuroshare replacement.

At the moment, there are 3 families of IOs:
    1. IO for reading closed manufacturers format (Spike2, Plexon, AlphaOmega, BlackRock, Axon, ...)
    2. IO for reading(/writing) formats from open source tools (KlustaKwik, Elan, WinEdr, WinWcp, PyNN, ...)
    3. IO for reading/writing neo structure in a neutral format (HDF5, .mat, ...) but with neo structure inside (NeoHDF5, NeoMatlab, ...)

Combining **1** for reading and **3** for writing is a good example of use : converting your datasets in a more standard format when you want to share/collaborate.


Introduction
===============


There is an intrinsic structure in the different Neo objects, that could be seen as a hierachy. See :ref:`core_page`.
The highest level object is the :class:`Block` object, which is the high level container able to encapsulate all the others.

A :class:`Block` is therefore a list of :class:`Segment` objects, that can, in some file formats, be accessed individually.
Depending on the file format, i.e. if it is streamable or not, the whole :class:`Block` may need to be loaded, but sometimes 
particular :class:`Segment` objects can be accessed individually.
Within a :class:`Segment`, the same hierarchical organisation applies.
A :class:`Segment` embeds several objects, such as :class:`SpikeTrain`  :class:`AnalogSignal`, :class:`AnaloSignalList`, :class:`EpochArray`, :class:`EventArray` (basically, all the different Neo objects).

Depending on the file format, these objects can sometimes be loaded separately, without the need to load the whole file.
If possible, a file IO therefore provides distinct methods allowing to load only particular objects that may be present in the file.
The basic idea of each IO file format is to have as much as possible read/write methods for the encapsulated individual objects, 
and otherwise, to provide a read/write method that will return the object at the highest level of hierarchy (by default, a :class:`Block` or a :class:`Segment`).

You have to anderstand that the neo.io API is a balanced between a full flexibility for the user (all read_XXX are enable) and a simple, clean and easy code (few read_XXX are enable).
So not all IOs offer the full flexibility for reading partilly data file.

One format = one class
===============================

The basic syntax is as follows. If you want to load a file format that is implemented in a generic MyFormatIO class::

    >>> from neo.io import MyFormatIO
    >>> reader = MyFormatIO("myfile.dat")

Supported object // readable object
====================================

To know what types of object are supported by this io interface::

    >>> file.supported_objects
    [Segment , AnalogSignal , SpikeTrain, Event, Spike ]

Supported objects does not mean objects that you can read directly. For instance, many formats support AnalogSignal
but you can't access them directly. To access your AnalogSignal, you must read a Segment with the **cascade** set to
True::

    >>> seg = reader.read_segment(cascade=True)
    >>> print seg.analogsignals
    >>> print seg.analogsignals[0]

To get a list of directly readable objects ::

    >>> reader.readable_objects
    [Segment]

The first element of the previous list is the highest level for reading the file. This mean that the IO have *read_segment* method::
    >>> seg = reader.read_segment()
    >>> type(seg)
    neo.core.Segment


All IOs have a read() method that return a block event if Segment is not in **readable_objects** or **supported_objects**::

    >>> bl = reader.read()
    >>> print bl.segments[0]
    neo.core.Segment


lazy and cascade
================================

In some case you may not want to load everything in memory because it could be to big.
You play with:

  * lazy=True/False. With lazy =True all array will have a size of zero.
  * cascade=True/False. With cascade=False only one object is readed, *one_to_many* and *many_to_many* relationship are readed.


Example::

    >>> seg = reader.read_segment(lazy = False, cascade=True)
    >>> print seg.analogsignals[0].shape #
    >>> seg = reader.read_segment(lazy = True, cascade=True)
    >>> print seg.analogsignals[0].shape # this is zero, the AnalogSIgnal is empty
    >>> seg = reader.read_segment(lazy = False, cascade=False)
    >>> print len(seg.analogsignals) # it is zeros

In the first case segment and subhierachy is read and all analosignal are loaded.
In second case segment and subhierachy is read but analogsignal are empty (size 0).
In the third case only AnalogSignal are not readed at all.

This simple mecanism is similar to reading a header (lazy =True, cascade = True) because everything is readed
execpt what is heavy.



Details of API
================

The neo.io API is designed to be simple and intuitive:
    - each file format has its IO classes (for example for Spike2 files you have a Spike2IO class)
    - each class inherits from the BaseIO class
    - each io class can read or write directly one or several neo objects (for example Segment, Block, ...): **readable_objects** or **writable_objects**
    - each io class support part of the neo.core hierachy not necessary all part: **supported_objects**
    - each io class have a read() method that return a Block even if there is only one Segment and one AnalogSignal inside.
    - each io is able to do a *lazy* load = all attribute are read execpet numpy.array (if lazy=True) _data_description attrbute is added.
    - each io is able to do a *cascade* load = if True all children object are also loaded
    - each io render object (and subojects) with all there necessary attributes 
    - each io can freely had remcommended attributs (and more) in _annotations dict of object.



List of implemented formats
=================================

.. automodule:: neo.io


If you want to developp your own IO
=====================================

See :ref:`io_dev_guide` for implementation of a new IO.

