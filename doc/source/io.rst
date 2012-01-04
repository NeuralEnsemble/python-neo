******
Neo IO
******

.. currentmodule:: neo

Preamble
========


The Neo :mod:`io` module aims to provide an exhaustive way of loading and saving several widely used data formats in electrophysiology.
The more these heterogeneous formats are supported, the easier it will be to manipulate them as Neo objects in a similar way.
Therefore the IO set of classes propose a simple and flexible IO API that fits many format specifications.
It is not only file-oriented, it can also read/write objects from a database.

:mod:`neo.io` can be seen as a *pure-Python* and open-source Neuroshare replacement.

At the moment, there are 3 families of IO modules:
    1. for reading closed manufacturers' formats (Spike2, Plexon, AlphaOmega, BlackRock, Axon, ...)
    2. for reading(/writing) formats from open source tools (KlustaKwik, Elan, WinEdr, WinWcp, PyNN, ...)
    3. for reading/writing Neo structure in neutral formats (HDF5, .mat, ...) but with Neo structure inside (NeoHDF5, NeoMatlab, ...)

Combining **1** for reading and **3** for writing is a good example of use: converting your datasets
to a more standard format when you want to share/collaborate.


Introduction
============

There is an intrinsic structure in the different Neo objects, that could be seen as a hierachy with cross-links. See :doc:`core`.
The highest level object is the :class:`Block` object, which is the high level container able to encapsulate all the others.

A :class:`Block` is therefore a list of :class:`Segment` objects, that can, in some file formats, be accessed individually.
Depending on the file format, i.e. if it is streamable or not, the whole :class:`Block` may need to be loaded, but sometimes 
particular :class:`Segment` objects can be accessed individually.
Within a :class:`Segment`, the same hierarchical organisation applies.
A :class:`Segment` embeds several objects, such as :class:`SpikeTrain`,
:class:`AnalogSignal`, :class:`AnaloSignalArray`, :class:`EpochArray`, :class:`EventArray`
(basically, all the different Neo objects).

Depending on the file format, these objects can sometimes be loaded separately, without the need to load the whole file.
If possible, a file IO therefore provides distinct methods allowing to load only particular objects that may be present in the file.
The basic idea of each IO file format is to have, as much as possible, read/write methods for the individual encapsulated objects, 
and otherwise to provide a read/write method that will return the object at the highest level of hierarchy
(by default, a :class:`Block` or a :class:`Segment`).

The :mod:`neo.io` API is a balance between full flexibility for the user (all :meth:`read_XXX` methods are enabled)
and simple, clean and understandable code for the developer (few :meth:`read_XXX` methods are enabled).
This means that not all IOs offer the full flexibility for partial reading of data files.

One format = one class
======================

The basic syntax is as follows. If you want to load a file format that is implemented in a generic :class:`MyFormatIO` class::

    >>> from neo.io import MyFormatIO
    >>> reader = MyFormatIO(filename = "myfile.dat")

you can replace :class:`MyFormatIO` by any implemented class, see :ref:`list_of_io`

Modes
======

IO can be based on file, directory, database or fake
This is describe in mode attribute of the IO class.

    >>> from neo.io import MyFormatIO
    >>> print MyFormatIO.mode
    'file'


For *file* mode the *filename* keyword argument is necessary.
For *directory* mode the *dirname* keyword argument is necessary.

Ex:
    >>> reader = io.PlexonIO(filename='File_plexon_1.plx')
    >>> reader = io.TdtIO(dirname='aep_05')


Supported objects/readable objects
==================================

To know what types of object are supported by a given IO interface::

    >>> MyFormatIO.supported_objects
    [Segment , AnalogSignal , SpikeTrain, Event, Spike]

Supported objects does not mean objects that you can read directly. For instance, many formats support :class:`AnalogSignal`
but don't allow them to be loaded directly, rather to access the :class:`AnalogSignal` objects, you must read a :class:`Segment`::

    >>> seg = reader.read_segment()
    >>> print(seg.analogsignals)
    >>> print(seg.analogsignals[0])

To get a list of directly readable objects ::

    >>> MyFormatIO.readable_objects
    [Segment]

The first element of the previous list is the highest level for reading the file. This mean that the IO has a :meth:`read_segment` method::

    >>> seg = reader.read_segment()
    >>> type(seg)
    neo.core.Segment


All IOs have a read() method that returns a :class:`Block` object::

    >>> bl = reader.read()
    >>> print bl.segments[0]
    neo.core.Segment


Lazy and cascade options
========================

In some cases you may not want to load everything in memory because it could be too big.
For this scenario, two options are available:

  * ``lazy=True/False``. With ``lazy=True`` all arrays will have a size of zero, but all the metadata will be loaded. lazy_shape attribute is added to all object that
    inheritate Quantitities or numpy.ndarray (AnalogSignal, AnalogSignalArray, SpikeTrain)  and to object that have array like attributes (EpochArray, EventArray)
    In that cases, lazy_shape is a tuple that have the same shape with lazy=False.
  * ``cascade=True/False``. With ``cascade=False`` only one object is read (and *one_to_many* and *many_to_many* relationship are not read).

By default (if they are not specified), ``lazy=False`` and ``cascade=True``, i.e. all data is loaded.

Example cascade::
    >>> seg = reader.read_segment( cascade=True)
    >>> print(len(seg.analogsignals)) # this is N
    >>> seg = reader.read_segment(cascade=False)
    >>> print(len(seg.analogsignals)) # this is zero

Example lazy::
    >>> seg = reader.read_segment(lazy=False)
    >>> print(seg.analogsignals[0].shape) # this is N
    >>> seg = reader.read_segment(lazy=True)
    >>> print(seg.analogsignals[0].shape) # this is zero, the AnalogSignal is empty
    >>> print(seg.analogsignals[0].lazy_shape) # this is N


.. _neo_io_API:

Details of API
==============

The :mod:`neo.io` API is designed to be simple and intuitive:
    - each file format has an IO class (for example for Spike2 files you have a :class:`Spike2IO` class).
    - each IO class inherits from the :class:`BaseIO` class.
    - each IO class can read or write directly one or several Neo objects (for example :class:`Segment`, :class:`Block`, ...): see the :attr:`readable_objects` and :attr:`writable_objects` attributes of the IO class.
    - each IO class supports part of the :mod:`neo.core` hierachy, though not necessarily all of it (see :attr:`supported_objects`).
    - each IO class has a :meth:`read()` method that returns a :class:`Block` even if there is only one :class:`Segment` and one :class:`AnalogSignal` inside.
    - each IO is able to do a *lazy* load: all metadata (e.g. :attr:`sampling_rate`) are read, but not the actual numerical data. lazy_shape attribute is added to provide information on real size.
    - each IO is able to do a *cascade* load: if ``True`` (default) all child objects are loaded, otherwise only the top level object is loaded.
    - each IO is able to save and load all required attributes (metadata) of the objects it supports. 
    - each IO can freely add user-defined or manufacturer-defined metadata to the :attr:`annotations` attribute of an object.


.. _list_of_io:

List of implemented formats
===========================

.. automodule:: neo.io


If you want to develop your own IO
==================================

See :doc:`io_developers_guide` for information on how to implement of a new IO.

