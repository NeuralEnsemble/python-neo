.. _io_dev_guide:

********************
IO developers' guide
********************


.. _io_guiline:

Guidelines for IO implementation
================================

Receipe to develop an IO module for a new data format:
    1. Fully understand the object model. See :doc:`core`. If in doubt ask the `mailing list`_.
    2. Fully understand :mod:`neo.io.exampleio`, It is a fake IO to explain the API. If in doubt ask the list.
    3. Copy/paste ``exampleio.py`` and choose clear file and class names for your IO.
    4. Decide which **supported objects** and **readable objects** your IO will deal with. This is the crucial point.
    5. Implement all methods :meth:`read_XXX` related to **readable objects**.
    6. Optional: If your IO supports reading multiple blocks from one file, implement a :meth:`read_all_blocks` method.
    7. Do not forget all lazy and cascade combinations.
    8. Optional: Support loading lazy objects by implementing a :meth:`load_lazy_object` method and / or lazy cascading by
       implementing a :meth:`load_lazy_cascade` method.
    9. Write good docstrings. List dependencies, including minimum version numbers.
    10. Add your class to :mod:`neo.io.__init__`. Keep the import inside try/except for dependency reasons.
    11. Contact the Neo maintainers to put sample files for testing on the G-Node server (write access is not public).
    12. Write tests in ``neo/test/io/test_xxxxxio.py``. You must at least pass the standard tests (inherited from :class:`BaseTestIO`).
    13. Commit or send a patch only if all tests pass.

Miscellaneous
=============

    * If you want to use logging in your IO, please use the standard
      :mod:`logging` module: create your logger using
      ``log = logging.getLogger('neo.io.my_io_name')`` or
      ``log = logging.getLogger(__name__)``
    * If your IO supports several version of a format (like ABF1, ABF2), upload to G-node test file repository all file version possible. (for utest coverage).
    * :py:func:`neo.core.Block.create_many_to_one_relationship` offers a utility to complete the hierachy when all one-to-many relationships have been created.
    * :py:func:`neo.io.tools.populate_RecordingChannel` offers a utility to
      create inside a :class:`Block` all :class:`RecordingChannel` objects and links to :class:`AnalogSignal`, :class:`SpikeTrain`, ...
    * In the docstring, explain where you obtained the file format specification if it is a closed one.
    * If your IO is based on a database mapper, keep in mind that the returned object MUST be detached,
      because this object can be written to another url for copying.
    
Advanced lazy loading
=====================

If your IO supports a format that might take a long time to load or require lots of memory, consider implementing one or both of the following methods to
enable advanced lazy loading:

* ``load_lazy_object(self, obj)``: This method takes a lazily loaded object and returns the corresponding fully loaded object.
  It does not set any links of the newly loaded object (e.g. the segment attribute of a SpikeTrain). The information needed to fully load the
  lazy object should usually be stored in the IO object (e.g. in a dictionary with lazily loaded objects as keys and the address
  in the file as values).
* ``load_lazy_cascade(self, address, lazy)``: This method takes two parameters: The information required by your IO to load an object and a boolean that
  indicates if data objects should be lazy loaded (in the same way as with regular :meth:`read_XXX` methods). The method should return a loaded
  objects, including all the links for one-to-many and many-to-many relationships (lists of links should be replaced by ``LazyList`` objects,
  see below).

  To implement lazy cascading, your read methods need to react when a user calls them with the ``cascade`` parameter set to ``lazy``.
  In this case, you have to replace all the link lists of your loaded objects with instances of :class:`neo.io.tools.LazyList`. Instead
  of the actual objects that your IO would load at this point, fill the list with items that ``load_lazy_cascade`` needs to load the
  object.

  Because the links of objects can point to previously loaded objects, you need to cache all loaded objects in the IO. If :meth:`load_lazy_cascade`
  is called with the address of a previously loaded object, return the object instead of loading it again. Also, a call to :meth:`load_lazy_cascade`
  might require you to load additional objects further up in the hierarchy. For example, if a :class:`SpikeTrain` is accessed through a
  :class:`Segment`, its :class:`Unit` and the :class:`RecordingChannelGroup` of the :class:`Unit` might have to be loaded at that point as well
  if they have not been accessed before.

  Note that you are free to restrict lazy cascading to certain objects. For example, you could use the ``LazyList`` only for the ``analogsignals``
  property of :class:`Segment` and :class:`RecordingChannel` objects and load the rest of file immediately.

Tests
=====

:py:class:`neo.test.io.commun_io_test.BaseTestIO` provide standard tests.
To use these you need to upload some sample data files at the `G-Node portal`_. They will be publicly accessible for testing Neo.
These tests:

  * check the compliance with the schema: hierachy, attribute types, ...
  * check if the IO respects the *lazy* and *cascade* keywords.
  * For IO able to both write and read data, it compares a generated dataset with the same data after a write/read cycle.

The test scripts download all files from the `G-Node portal`_ and store them locally in ``neo/test/io/files_for_tests/``.
Subsequent test runs use the previously downloaded files, rather than trying to download them each time.

Here is an example test script taken from the distribution: ``test_axonio.py``:

.. literalinclude:: ../../neo/test/iotest/test_axonio.py



ExampleIO
=========

.. autoclass:: neo.io.ExampleIO

Here is the entire file:

.. literalinclude:: ../../neo/io/exampleio.py


.. _`mailing list`: http://groups.google.com/group/neuralensemble
.. _G-node portal: https://portal.g-node.org/neo/
