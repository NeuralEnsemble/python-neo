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

At the moment, there are 3 families of IO modules:
    1. for reading closed manufacturers' formats (Spike2, Plexon, AlphaOmega, BlackRock, Axon, ...)
    2. for reading(/writing) formats from open source tools (KlustaKwik, Elan, WinEdr, WinWcp, ...)
    3. for reading/writing Neo structure in neutral formats (HDF5, .mat, ...) but with Neo structure inside (NeoHDF5, NeoMatlab, ...)

Combining **1** for reading and **3** for writing is a good example of use: converting your datasets
to a more standard format when you want to share/collaborate.


Introduction
============

There is an intrinsic structure in the different Neo objects, that could be seen as a hierachy with cross-links. See :doc:`core`.
The highest level object is the :class:`Block` object, which is the high level container able to encapsulate all the others.

A :class:`Block` has therefore a list of :class:`Segment` objects, that can, in some file formats, be accessed individually.
Depending on the file format, i.e. if it is streamable or not, the whole :class:`Block` may need to be loaded, but sometimes
particular :class:`Segment` objects can be accessed individually.
Within a :class:`Segment`, the same hierarchical organisation applies.
A :class:`Segment` embeds several objects, such as :class:`SpikeTrain`,
:class:`AnalogSignal`, :class:`IrregularlySampledSignal`, :class:`Epoch`, :class:`Event`
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
    >>> reader = MyFormatIO(filename="myfile.dat")

you can replace :class:`MyFormatIO` by any implemented class, see :ref:`list_of_io`

Modes
======

IO can be based on a single file, a directory containing files, or a database.
This is described in the :attr:`mode` attribute of the IO class.

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


All IOs have a read() method that returns a list of :class:`Block` objects (representing the whole content of the file)::

    >>> bl = reader.read()
    >>> print bl[0].segments[0]
    neo.core.Segment

    
Read a time slice of Segment
============================

Some object support ``time_slice`` arguments in ``read_segment()``.
This is usefull to read only a subset of a dataset clipped in time.
By default  ``time_slice=None`` meaning load eveything.

This read everything::

    seg = reader.load_segment(time_slice=None)
    shape0 = seg.analogsignals[0].shape # total shape
    shape1 = seg.spiketrains[0].shape  # total shape

    
This read only the first 5 seconds::

    seg = reader.load_segment(time_slice=(0*pq.s, 5.*pq.s))
    shape0 = seg.analogsignals[0].shape # more small shape
    shape1 = seg.spiketrains[0].shape  # more small shape


Lazy option and proxy object
============================

In some cases you may not want to load everything in memory because it could be too big.
For this scenario, some IOs implement ``lazy=True/False``. 
Since neo 0.7, a new lazy sytem have been added for some IOs (all IOs that inherits rawio).
To know if a class supports lazy mode use ``ClassIO.support_lazy``.

With ``lazy=True`` all data object (AnalogSignal/SpikeTrain/Event/Epoch) are replace by 
ProxyObjects (AnalogSignalProxy/SpikeTrainProxy/EventProxy/EpochProxy).


By default (if not specified), ``lazy=False``, i.e. all data is loaded.

Theses ProxyObjects contain metadata (name, sampling_rate, id, ...) so they can be inspected
but they do not contain any array-like data.
All ProxyObjects contain a ``load()`` method to postpone the real load of array like data.

Further more the  ``load()`` method have a ``time_slice`` arguments to load only a slice
from the file. So the consumption of memory can be finely controlled. 


Here 2 examples that do read a dataset,  load triggers a do trigger averaging on signals.

The first example is without lazy, so it consume more memory::
    
    lim0, lim1 = -500*pq.ms, +1500*pq.ms
    seg = reader.read_segment(lazy=False)
    triggers = seg.events[0]
    anasig = seg.analogsignals[0]  # here anasig contain the whole recording in memory
    all_sig_chunks = []
    for t in triggers.times:
        t0, t1 = (t + lim0), (t + lim1)
        anasig_chunk = anasig.time_slice(t0, t1)
        all_sig_chunks.append(anasig_chunk)
    apply_my_fancy_average(all_sig_chunks)
    
The second example use lazy so it consume fewer memory::

    lim0, lim1 = -500*pq.ms, +1500*pq.ms
    seg = reader.read_segment(lazy=True)
    triggers = seg.events[0].load(time_slice=None)  # this load all trigers in memory
    anasigproxy = seg.analogsignals[0]  # this is a proxy
    all_sig_chunks = []
    for t in triggers.times:
        t0, t1 = (t + lim0), (t + lim1)
        anasig_chunk = anasigproxy.load(time_slice=(t0, t1))  # here real data are loaded
        all_sig_chunks.append(anasig_chunk)
    apply_my_fancy_average(all_sig_chunks)

In addition to ``time_slice`` AnalogSignalProxy contains ``channel_indexes`` arguments.
This control also slicing in channel. This is usefull in case the channel count is very high.

 .. TODO: add something about magnitude mode when implemented for all objects.

In this example, we read only 3 selected channels::

    seg = reader.read_segment(lazy=True)
    anasig = seg.analogsignals[0].load(time_slice=None, channel_indexes=[0, 2, 18])


.. _neo_io_API:

Details of API
==============

The :mod:`neo.io` API is designed to be simple and intuitive:
    - each file format has an IO class (for example for Spike2 files you have a :class:`Spike2IO` class).
    - each IO class inherits from the :class:`BaseIO` class.
    - each IO class can read or write directly one or several Neo objects (for example :class:`Segment`, :class:`Block`, ...): see the :attr:`readable_objects` and :attr:`writable_objects` attributes of the IO class.
    - each IO class supports part of the :mod:`neo.core` hierachy, though not necessarily all of it (see :attr:`supported_objects`).
    - each IO class has a :meth:`read()` method that returns a list of :class:`Block` objects. If the IO only supports :class:`Segment` reading, the list will contain one block with all segments from the file.
    - each IO class that supports writing has a :meth:`write()` method that takes as a parameter a list of blocks, a single block or a single segment, depending on the IO's :attr:`writable_objects`.
    - some IO are able to do a *lazy* load: all metadata (e.g. :attr:`sampling_rate`) are read, but not the actual numerical data.
    - each IO is able to save and load all required attributes (metadata) of the objects it supports.
    - each IO can freely add user-defined or manufacturer-defined metadata to the :attr:`annotations` attribute of an object.


If you want to develop your own IO
==================================

See :doc:`io_developers_guide` for information on how to implement a new IO.


.. _list_of_io:

List of implemented formats
===========================

.. automodule:: neo.io


Logging
=======

:mod:`neo` uses the standard Python :mod:`logging` module for logging.
All :mod:`neo.io` classes have logging set up by default, although not all classes produce log messages.
The logger name is the same as the full qualified class name, e.g. :class:`neo.io.hdf5io.NeoHdf5IO`.
By default, only log messages that are critically important for users are displayed, so users should not disable log messages unless they are sure they know what they are doing.
However, if you wish to disable the messages, you can do so::

    >>> import logging
    >>>
    >>> logger = logging.getLogger('neo')
    >>> logger.setLevel(100)

Some io classes provide additional information that might be interesting to advanced users.
To enable these messages, do the following::

    >>> import logging
    >>>
    >>> logger = logging.getLogger('neo')
    >>> logger.setLevel(logging.INFO)

It is also possible to log to a file in addition to the terminal::

    >>> import logging
    >>>
    >>> logger = logging.getLogger('neo')
    >>> handler = logging.FileHandler('filename.log')
    >>> logger.addHandler(handler)

To only log to the terminal::

    >>> import logging
    >>> from neo import logging_handler
    >>>
    >>> logger = logging.getLogger('neo')
    >>> handler = logging.FileHandler('filename.log')
    >>> logger.addHandler(handler)
    >>>
    >>> logging_handler.setLevel(100)

This can also be done for individual IO classes::

    >>> import logging
    >>>
    >>> logger = logging.getLogger('neo.io.hdf5io.NeoHdf5IO')
    >>> handler = logging.FileHandler('filename.log')
    >>> logger.addHandler(handler)

Individual IO classes can have their loggers disabled as well::

    >>> import logging
    >>>
    >>> logger = logging.getLogger('neo.io.hdf5io.NeoHdf5IO')
    >>> logger.setLevel(100)

And more detailed logging messages can be enabled for individual IO classes::

    >>> import logging
    >>>
    >>> logger = logging.getLogger('neo.io.hdf5io.NeoHdf5IO')
    >>> logger.setLevel(logging.INFO)

The default handler, which is used to print logs to the command line, is stored in :attr:`neo.logging_handler`.
This example changes how the log text is displayed::

    >>> import logging
    >>> from neo import logging_handler
    >>>
    >>> formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    >>> logging_handler.setFormatter(formatter)

For more complex logging, please see the documentation for the logging_ module.

.. note:: If you wish to implement more advanced logging as describe in the documentation for the logging_ module or elsewhere on the internet, please do so before calling any :mod:`neo` functions or initializing any :mod:`neo` classes.
          This is because the default handler is created when :mod:`neo` is imported, but it is not attached to the :mod:`neo` logger until a class that uses logging is initialized or a function that uses logging is called.
          Further, the handler is only attached if there are no handlers already attached to the root logger or the :mod:`neo` logger, so adding your own logger will override the default one.
          Additional functions and/or classes may get logging during bugfix releases, so code relying on particular modules not having logging may break at any time without warning.

.. _`logging`: http://docs.python.org/library/logging.html
