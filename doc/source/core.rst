********
Neo core
********

.. currentmodule:: neo.core

This figure shows the main data types in Neo, with the exception of the newly added ImageSequence and RegionOfInterest classes:

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
  * :py:class:`ImageSequence`: A three dimensional array representing a sequence of images.

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
    Contains :class:`Segment` and :class:`Group` objects.


Grouping/linking objects
------------------------

These objects express the relationships between data items, such as which signals
were recorded on which electrodes, which spike trains were obtained from which
membrane potential signals, etc. They contain references to data objects that
cut across the simple container hierarchy.

  * :py:class:`ChannelView`: A set of indices into :py:class:`AnalogSignal` objects,
    representing logical and/or physical recording channels.
    For spike sorting of extracellular signals, where spikes may be recorded on more than one
    recording channel, the :py:class:`ChannelView` can be used to reference the group of recording channels
    from which the spikes were obtained.

  * :py:class:`Group`: Can contain any of the data objects, views, or other groups,
    outside the hierarchy of the segment and block containers.
    A common use is to link the :class:`SpikeTrain` objects within a :class:`Block`,
    possibly across multiple Segments, that were emitted by the same neuron.

  * :py:class:`CircularRegionOfInterest`, :py:class:`RectangularRegionOfInterest` and :py:class:`PolygonRegionOfInterest`
    are three subclasses that link :class:`ImageSequence` objects to signals (:class:`AnalogSignal` objects)
    extracted from them.

For more details, see :doc:`grouping`.


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
    * :attr:`Block.groups`

These relationships are bi-directional, i.e. a child object can access its parent:

    * :attr:`Segment.block`
    * :attr:`AnalogSignal.segment`
    * :attr:`SpikeTrain.segment`
    * :attr:`Group.block`

Here is an example showing these relationships in use::

    from neo.io import AxonIO
    import urllib.request
    url = "https://web.gin.g-node.org/NeuralEnsemble/ephy_testing_data/raw/master/axon/File_axon_3.abf"
    filename = './test.abf'
    urllib.request.urlretrieve(url, filename)

    r = AxonIO(filename=filename)
    blocks = r.read() # read the entire file > a list of Blocks
    bl = blocks[0]
    print(bl)
    print(bl.segments) # child access
    for seg in bl.segments:
        print(seg)
        print(seg.block) # parent access


In some cases, a one-to-many relationship is sufficient. Here is a simple example with tetrodes, in which each tetrode has its own group.::

    from neo import Block, Group
    bl = Block()

    # the four tetrodes
    for i in range(4):
        group = Group(name='Tetrode %d' % i)
        bl.groups.append(group)

    # now we load the data and associate it with the created channels
    # ...

Now consider a more complex example: a 1x4 silicon probe, with a neuron on channels 0,1,2 and another neuron on channels 1,2,3.
We create a group for each neuron to hold the spiketrains for each spike sorting group together with
the channels on which that neuron spiked::

    bl = Block(name='probe data')

    # one group for each neuron
    view0 = ChannelView(recorded_signals, index=[0, 1, 2])
    unit0 = Group(view0, name='Group 0')
    bl.groups.append(unit0)

    view1 = ChannelView(recorded_signals, index=[1, 2, 3])
    unit1 = Group(view1, name='Group 1')
    bl.groups.append(unit1)

    # now we add the spiketrains from Unit 0 to unit0
    # and add the spiketrains from Unit 1 to unit1
    # ...


Now each putative neuron is represented by a :class:`Group` containing the spiktrains of that neuron
and a view of the signal selecting only those channels from which the spikes were obtained.


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

.. note:: This figure do not include :class:`ChannelView` and :class:`RegionOfInterest`.

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

Array Annotations
-----------------

Next to "regular" annotations there is also a way to annotate arrays of values
in order to create annotations with one value per data point. Using this feature,
called Array Annotations, the consistency of those annotations with the actual data
is ensured.
Apart from adding those on object construction, Array Annotations can also be added
using the :meth:`array_annotate` method provided by all Neo data objects, e.g.::

    >>> sptr = SpikeTrain(times=[1, 2, 3]*pq.s, t_stop=3*pq.s)
    >>> sptr.array_annotate(index=[0, 1, 2], relevant=[True, False, True])
    >>> print(sptr.array_annotations)
    {'index': array([0, 1, 2]), 'relevant': array([ True, False,  True])}

Since Array Annotations may be written to a file or database, there are some
limitations on the data types of arrays: they must be 1-dimensional (i.e. not nested)
and contain the same types as annotations:

    ``integer``, ``float``, ``complex``, ``Quantity``, ``string``, ``date``, ``time`` and ``datetime``.
