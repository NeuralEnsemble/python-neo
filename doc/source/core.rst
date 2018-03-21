********
Neo core
********

.. currentmodule:: neo.core

This figure shows the main data types in Neo:

.. image:: images/base_schematic.png
   :height: 500 px
   :alt: Illustration of the main Neo data types
   :align: center

Neo objects fall into three categories: data objects, container objects and grouping objects.

Data objects
------------

These objects directly represent data as arrays of numerical values with
associated metadata (units, sampling frequency, etc.).

  * :py:class:`AnalogSignal`: A regular sampling of a single- or multi-channel continuous analog signal.
  * :py:class:`IrregularlySampledSignal`: A non-regular sampling of a single- or multi-channel continuous analog signal.
  * :py:class:`SpikeTrain`: A set of action potentials (spikes) emitted by the same unit in a period of time (with optional waveforms).
  * :py:class:`Event`: An array of time points representing one or more events in the data.
  * :py:class:`Epoch`: An array of time intervals representing one or more periods of time in the data.


Container objects
-----------------

There is a simple hierarchy of containers:

  * :py:class:`Segment`: A container for heterogeneous discrete or continous data sharing a common
    clock (time basis) but not necessarily the same sampling rate, start time or end time.
    A :py:class:`Segment` can be considered as equivalent to a "trial", "episode", "run",
    "recording", etc., depending on the experimental context.
    May contain any of the data objects.
  * :py:class:`Block`: The top-level container gathering all of the data, discrete and continuous,
    for a given recording session.
    Contains :class:`Segment`, :class:`Unit` and :class:`ChannelIndex` objects.


Grouping objects
----------------

These objects express the relationships between data items, such as which signals
were recorded on which electrodes, which spike trains were obtained from which
membrane potential signals, etc. They contain references to data objects that
cut across the simple container hierarchy.

  * :py:class:`ChannelIndex`: A set of indices into :py:class:`AnalogSignal` objects,
    representing logical and/or physical recording channels. This has two uses:

      1. for linking :py:class:`AnalogSignal` objects recorded from the same (multi)electrode
         across several :py:class:`Segment`\s.
      2. for spike sorting of extracellular signals, where spikes may be recorded on more than one
         recording channel, and the :py:class:`ChannelIndex` can be used to associate each
         :py:class:`Unit` with the group of recording channels from which it was obtained.

  * :py:class:`Unit`: links the :class:`SpikeTrain` objects within a :class:`Block`,
    possibly across multiple Segments, that were emitted by the same cell.
    A :class:`Unit` is linked to the :class:`ChannelIndex` object from which the spikes were detected.


NumPy compatibility
===================

Neo data objects inherit from :py:class:`Quantity`, which in turn inherits from NumPy
:py:class:`ndarray`. This means that a Neo :py:class:`AnalogSignal` is also a :py:class:`Quantity`
and an array, giving you access to all of the methods available for those objects.

For example, you can pass a :py:class:`SpikeTrain` directly to the :py:func:`numpy.histogram`
function, or an :py:class:`AnalogSignal` directly to the :py:func:`numpy.std` function.

If you want to get a numpy.ndarray you use magnitude and rescale from quantities::

   >>> np_sig = neo_analogsignal.rescale('mV').magnitude
   >>> np_times = neo_analogsignal.times.rescale('s').magnitude

Relationships between objects
=============================

Container objects like :py:class:`Block` or :py:class:`Segment` are gateways to
access other objects. For example, a :class:`Block` can access a :class:`Segment`
with::

    >>> bl = Block()
    >>> bl.segments
    # gives a list of segments

A :class:`Segment` can access the :class:`AnalogSignal` objects that it contains with::

    >>> seg = Segment()
    >>> seg.analogsignals
    # gives a list of AnalogSignals

In the :ref:`neo_diagram` below, these *one to many* relationships are represented by cyan arrows.
In general, an object can access its children with an attribute *childname+s* in lower case, e.g.

    * :attr:`Block.segments`
    * :attr:`Segments.analogsignals`
    * :attr:`Segments.spiketrains`
    * :attr:`Block.channel_indexes`

These relationships are bi-directional, i.e. a child object can access its parent:

    * :attr:`Segment.block`
    * :attr:`AnalogSignal.segment`
    * :attr:`SpikeTrain.segment`
    * :attr:`ChannelIndex.block`

Here is an example showing these relationships in use::

    from neo.io import AxonIO
    import urllib
    url = "https://portal.g-node.org/neo/axon/File_axon_3.abf"
    filename = './test.abf'
    urllib.urlretrieve(url, filename)

    r = AxonIO(filename=filename)
    bl = r.read() # read the entire file > a Block
    print(bl)
    print(bl.segments) # child access
    for seg in bl.segments:
        print(seg)
        print(seg.block) # parent access


In some cases, a one-to-many relationship is sufficient. Here is a simple example with tetrodes, in which each tetrode has its own group.::

    from neo import Block, ChannelIndex
    bl = Block()

    # the four tetrodes
    for i in range(4):
        chx = ChannelIndex(name='Tetrode %d' % i,
                           index=[0, 1, 2, 3])
        bl.channelindexes.append(chx)

    # now we load the data and associate it with the created channels
    # ...

Now consider a more complex example: a 1x4 silicon probe, with a neuron on channels 0,1,2 and another neuron on channels 1,2,3. We create a group for each neuron to hold the :class:`Unit` object associated with this spike sorting group. Each group also contains the channels on which that neuron spiked. The relationship is many-to-many because channels 1 and 2 occur in multiple groups.::

    bl = Block(name='probe data')

    # one group for each neuron
    chx0 = ChannelIndex(name='Group 0',
                        index=[0, 1, 2])
    bl.channelindexes.append(chx0)

    chx1 = ChannelIndex(name='Group 1',
                        index=[1, 2, 3])
    bl.channelindexes.append(chx1)

    # now we add the spiketrain from Unit 0 to chx0
    # and add the spiketrain from Unit 1 to chx1
    # ...

Note that because neurons are sorted from groups of channels in this situation, it is natural that the :py:class:`ChannelIndex` contains a reference to the :py:class:`Unit` object.
That unit then contains references to its spiketrains. Also note that recording channels can be
identified by names/labels as well as, or instead of, integer indices.


See :doc:`usecases` for more examples of how the different objects may be used.

.. _neo_diagram:

Neo diagram
===========

Object:
  * With a star = inherits from :class:`Quantity`
Attributes:
  * In red = required
  * In white = recommended
Relationship:
  * In cyan = one to many
  * In yellow = properties (deduced from other relationships)


.. image:: images/simple_generated_diagram.png
    :width: 750 px

:download:`Click here for a better quality SVG diagram <./images/simple_generated_diagram.svg>`

For more details, see the :doc:`api_reference`.

Initialization
==============

Neo objects are initialized with "required", "recommended", and "additional" arguments.

    - Required arguments MUST be provided at the time of initialization. They are used in the construction of the object.
    - Recommended arguments may be provided at the time of initialization. They are accessible as Python attributes. They can also be set or modified after initialization.
    - Additional arguments are defined by the user and are not part of the Neo object model. A primary goal of the Neo project is extensibility. These additional arguments are entries in an attribute of the object: a Python dict called :py:attr:`annotations`. 
      Note : Neo annotations are not the same as the *__annotations__* attribute introduced in Python 3.6.

Example: SpikeTrain
-------------------

:py:class:`SpikeTrain` is a :py:class:`Quantity`, which is a NumPy array containing values with physical dimensions. The spike times are a required attribute, because the dimensionality of the spike times determines the way in which the :py:class:`Quantity` is constructed.

Here is how you initialize a :py:class:`SpikeTrain` with required arguments::

    >>> import neo
    >>> st = neo.SpikeTrain([3, 4, 5], units='sec', t_stop=10.0)
    >>> print(st)
    [ 3.  4.  5.] s

You will see the spike times printed in a nice format including the units.
Because `st` "is a" :py:class:`Quantity` array with units of seconds, it absolutely must have this information at the time of initialization. You can specify the spike times with a keyword argument too::

    >>> st = neo.SpikeTrain(times=[3, 4, 5], units='sec', t_stop=10.0)

The spike times could also be in a NumPy array.

If it is not specified, :attr:`t_start` is assumed to be zero, but another value can easily be specified::

    >>> st = neo.SpikeTrain(times=[3, 4, 5], units='sec', t_start=1.0, t_stop=10.0)
    >>> st.t_start
    array(1.0) * s


Recommended attributes must be specified as keyword arguments, not positional arguments.


Finally, let's consider "additional arguments". These are the ones you define for your experiment::

    >>> st = neo.SpikeTrain(times=[3, 4, 5], units='sec', t_stop=10.0, rat_name='Fred')
    >>> print(st.annotations)
    {'rat_name': 'Fred'}

Because ``rat_name`` is not part of the Neo object model, it is placed in the dict :py:attr:`annotations`. This dict can be modified as necessary by your code.

Annotations
-----------

As well as adding annotations as "additional" arguments when an object is
constructed, objects may be annotated using the :meth:`annotate` method
possessed by all Neo core objects, e.g.::

    >>> seg = Segment()
    >>> seg.annotate(stimulus="step pulse", amplitude=10*nA)
    >>> print(seg.annotations)
    {'amplitude': array(10.0) * nA, 'stimulus': 'step pulse'}

Since annotations may be written to a file or database, there are some
limitations on the data types of annotations: they must be "simple" types or
containers (lists, dicts, tuples, NumPy arrays) of simple types, where the simple types
are ``integer``, ``float``, ``complex``, ``Quantity``, ``string``, ``date``, ``time`` and
``datetime``.
