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
    6. Do not forget all : lasy and cascade combination.
    7. Write good docstrings. List dependencies, including minimum version numbers.
    8. Add your class to :mod:`neo.io.__init__`. Keep the import inside try/except for dependency reasons.
    9. Contact the Neo maintainers to put sample files for testing on the G-Node server (write access is not public).
    10. Write tests in ``neo/test/io/test_xxxxxio.py``. You must at least pass the standard tests (inherited from :class:`BaseTestIO`).
    11. Commit or send a patch only if all tests pass.


Miscellaneous
=============

Notes:
    * if your IO supports several version of a format (like ABF1, ABF2), upload to G-node test file repository all file version possible. (for utest coverage).
    * :py:func:`neo.io.tools.create_many_to_one_relationship` offers a utility to complete the hierachy when all one-to-many relationships have been created.
    * :py:func:`neo.io.tools.populate_RecordingChannel` offers a utility to
      create inside a :class:`Block` all :class:`RecordingChannel` objects and links to :class:`AnalogSignal`, :class:`SpikeTrain`, ...
    * In the docstring, explain where you obtained the file format specification if it is a closed one.
    * If your IO is based on a database mapper, keep in mind that the returned object MUST be detached,
      because this object can be written to another url for copying.
    


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

.. literalinclude:: ../../neo/test/io/test_axonio.py



ExampleIO
=========

.. autoclass:: neo.io.ExampleIO

Here is the entire file:

.. literalinclude:: ../../neo/io/exampleio.py


.. _`mailing list`: http://groups.google.com/group/neuralensemble
.. _G-node portal: https://portal.g-node.org/neo/