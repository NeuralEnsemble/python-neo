********
Neo core
********

.. currentmodule:: neo

Introduction
============

Objects in Neo represent neural data and collections of data. Neo objects fall
into three categories: data objects, container objects and grouping objects.

Data objects
------------

These objects directly represent data as arrays of numerical values with
associated metadata (e.g. units, sampling frequency, etc.)

:py:class:`AnalogSignal`:
    A representation of a continuous, analog signal acquired at time ``t_start`` at a certain sampling rate.

:py:class:`AnalogSignalArray`:
    A representation of multiple continuous, analog signals, all acquired the the same time and with the same sampling rate. This representation (as a 2D NumPy array) may be more efficient for subsequent analysis than the equivalent list of individual :py:class:`AnalogSignal` objects.

:py:class:`SpikeTrain`:
    An ensemble of action potentials (spikes) emitted by the same unit in a period of time.

:py:class:`Event` and :py:class:`EventArray`:
    A time point representng an event in the data, or an array of such time points.

:py:class:`Epoch` and :py:class:`EpochArray`:
    An interval of time representing a period of time in the data, or an array of such intervals.

Container objects
-----------------

There is a simple hierarchy of containers:

:py:class:`Segment`:
    A container for heterogeneous discrete or continous data sharing a common clock (time basis) but not necessarily the same sampling rate, start time or end time. A :py:class:`Segment` can be considered as equivalent to a "trial", "episode", "run", "recording", etc., depending on the experimental context. May contain any of the data objects together with :class:`RecordingChannel` and :class:`Unit` objects.

:py:class:`Block`:
    The top-level container gathering all of the data, discrete and continuous, for a given recording session. Contains :class:`Segment` and :class:`RecordingChannelGroup` objects.

Grouping objects
----------------

These objects express the relationships between data items, such as which signals
were recorded on which electrodes, which spike trains were obtained from which
membrane potential signals, etc. They contain references to data objects that
cut across the simple container hierarchy.

:py:class:`RecordingChannel`:
    Links :py:class:`AnalogSignal`, :py:class:`SpikeTrain`.
    objects that come from the same logical and/or physical channel inside a :py:class:`Block` across  several :py:class:`Segment` .

:py:class:`RecordingChannelGroup`:
    A group for associated :py:class:`RecordingChannel` objects. This has several possible uses: 
      * The main one is for linking several :py:class:`AnalogSignalArray` across several  :py:class:`Segment` inside a  :py:class:`Block`.
      * A second use for multielectrode arrays, spikes may be recorded on more than one recording channel, 
        and so the :py:class:`RecordingChannelGroup` can be used to associate each :py:class:`Unit` with the
        group of recording channels on which it was calculated (Example a tetrode).
      * A third use and flexible use is for grouping several :py:class:`RecordingChannel`. There are many case for that.
        For instance for intracellular recording, it is common to record both membrane potentials and currents at the same time, 
        so each :py:class:`RecordingChannelGroup` may correspond to the particular property that is being recorded. For Multi Electrode Array,
        :py:class:`RecordingChannelGroup` is used to gather all :py:class:`RecordingChannel` of the same MEA.
        
:py:class:`Unit`:
    It regroups all the :class:`SpikeTrain` objects that were emitted by a neuron during a :class:`Block`. 
    A :class:`Unit` is linked to :class:`RecordingChannelGroup` objects from which it was detected.
    This replaces the :class:`Neuron` class in the previous version of Neo (V1).

.. image:: images/base_schematic.png
   :height: 500 px
   :alt: Neo : Neurotools/OpenElectrophy shared base architecture 
   :align: center

.. todo:: fill in missing descriptions above

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
    # gives a list a AnalogSignals
    
In the diagram below, these *one to many* relationships are represented by cyan arrows.
In general, an object can access its children with an attribute *childname+s* in lower case, e.g.

    * :attr:`Block.segments`
    * :attr:`Segments.analogsignals`
    * :attr:`Segments.spiketrains`
    * :attr:`Block.recordingchannelgroups`

These relationships are bi-directional, i.e. a child object can access its parent:

    * :attr:`Segment.block`
    * :attr:`AnalogSignal.segment`
    * :attr:`SpikeTrains.segment`
    * :attr:`RecordingChannelGroup.block`

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


On the diagram you can also see a magenta relationship. This is more tricky *many to many* relationship.
This relationship is between :py:class:`RecordingChannel` and :py:class:`RecordingChannelGroup`.
*Many to many* relationship means that :py:class:`RecordingChannelGroup` have a list *recordingchannels** 
that point to many :py:class:`RecordingChannel`, this is the intuitive and general case. But there also a
link :py:class:`RecordingChannel`.*recordingchannelgroups* to many  :py:class:`RecordingChannelGroup`.
this second and less intuitive link describe the fact that a :py:class:`RecordingChannel` can belong to
several groups.

An example for helping, take this case: I have a probe with 16 channel divided in 4 tetrodes.
I want 5 groups: one for describing the whole probe and four for each tetrodes::

    from neo import *
    bl = Block()
    
    # creating individual channel
    all_rc= [ ]
    for i in range(16):
        rc = RecordingChannel( index= i, name ='rc %d' %i)
        all_rc.append(rc)
    
    # global group
    rcg = RecordginChannelGroup( name = 'The whole probe')
    for rc in all_rc:
        rcg.recordingchannels.append(rc)
        rc.recordingchannelgroups.append(rcg)
    bl.recordingchannelgroups.append(rcg)
    
    # the four tetrodes
    for i in range(4):
        rcg = RecordginChannelGroup( name = 'Tetrode %d' % i )
        for rc in all_rc[i*4:(i+1)*4]:
            rcg.recordingchannels.append(rc)
            rc.recordingchannelgroups.append(rcg)
        bl.recordingchannelgroups.append(rcg)



See :ref:`use_cases_page` for more examples of how the different objects may be used.

.. _neo_diagram:


.. image:: images/simple_generated_diagram.png
    :width: 750 px

:download:`Click here for a better quality SVG diagram <./images/simple_generated_diagram.svg>`

For more details, see the :doc:`api_reference`.

    

Inheritance
===========

Some Neo objects (:py:class:`AnalogSignal`, :py:class:`SpikeTrain`, :py:class:`AnalogSignalArray`) inherit from :py:class:`Quantity`, which in turn inherits from NumPy :py:class:`ndarray`. This means that a Neo :py:class:`AnalogSignal` actually is also a :py:class:`Quantity` and an array, giving you access to all of the methods available for those objects.

For example, you can pass a :py:class:`SpikeTrain` directly to the :py:func:`numpy.histogram` function, or an :py:class:`AnalogSignal` directly to the :py:func:`numpy.std` function.


Initialization
==============

Neo objects are initialized with "required", "recommended", and "additional" arguments.

    - Required arguments MUST be provided at the time of initialization. They are used in the construction of the object.
    - Recommended arguments may be provided at the time of initialization. They are accessible as Python attributes. They can also be set or modified after initialization.
    - Additional arguments are defined by the user and are not part of the Neo object model. A primary goal of the Neo project is extensibility. These additional arguments are entries in an attribute of the object: a Python dict called :py:attr:`annotations`.

Example: SpikeTrain
-------------------

:py:class:`SpikeTrain` is a :py:class:`Quantity`, which is a NumPy array with dimensionality. The spike times are a required attribute, because the dimensionality of the spike times determines the way in which the :py:class:`Quantity` is constructed.

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

.. note:: Note for developers: A glance at the underlying code reveals the implementation distinction between required and recommended attributes. Required attributes are set in :py:meth:`object.__new__`, while recommended attributes are set in :py:meth:`object.__init__`


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
containers (lists, dicts, NumPy arrays) of simple types, where the simple types
are ``integer``, ``float``, ``Quantity``, ``string``, ``date``, ``time`` and
``datetime``.
