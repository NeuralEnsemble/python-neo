.. _io_dev_guide:

********************
IO developers' guide
********************


.. _io_guiline:

Guidelines for IO implementation
================================

There are two ways to add a new IO module:
  * By directly adding a new IO class in a module within :mod:`neo.io`: the reader/writer will deal directly with Neo objects
  * By adding a RawIO class in a module within :mod:`neo.rawio`: the reader should work with raw buffers from the file and provide
    some internal headers for the scale/units/name/... 
    You can then generate an IO module simply by inheriting from your RawIO class and from :class:`neo.io.BaseFromRaw`

For read only classes, we encourage you to write a :class:`RawIO` class because it allows slice reading,
and is generally much quicker and easier (although only for reading) than implementing a full IO class.
For read/write classes you can mix the two levels neo.rawio for reading and neo.io for writing.

Recipe to develop an IO module for a new data format:
    1. Fully understand the object model. See :doc:`core`. If in doubt ask the `mailing list`_.
    2. Fully understand :mod:`neo.io.examplerawio`, It is a fake IO to explain the API. If in doubt ask the list.
    3. Copy/paste ``examplerawio.py`` and choose clear file and class names for your IO.
    4. implement all methods that **raise(NotImplementedError)** in :mod:`neo.rawio.baserawio`. Return None when the object is not supported (spike/waveform)
    5. Write good docstrings. List dependencies, including minimum version numbers.
    6. Add your class to :mod:`neo.rawio.__init__`. Keep imports inside ``try/except`` for dependency reasons.
    7. Create a class in :file:`neo/io/`
    8. Add your class to :mod:`neo.io.__init__`. Keep imports inside ``try/except`` for dependency reasons.
    9. Create an account at https://gin.g-node.org and deposit files in :file:`NeuralEnsemble/ephy_testing_data`.
    10. Write tests in :file:`neo/rawio/test_xxxxxrawio.py`. You must at least pass the standard tests (inherited from :class:`BaseTestRawIO`). See :file:`test_examplerawio.py`
    11. Write a similar test in :file:`neo.tests/iotests/test_xxxxxio.py`. See :file:`test_exampleio.py`
    12. Make a pull request when all tests pass.

Miscellaneous
=============

    * If your IO supports several versions of a format (like ABF1, ABF2), upload to the gin.g-node.org test file repository all file versions possible. (for test coverage).
    * :py:func:`neo.core.Block.create_many_to_one_relationship` offers a utility to complete the hierachy when all one-to-many relationships have been created.
    * In the docstring, explain where you obtained the file format specification if it is a closed one.
    * If your IO is based on a database mapper, keep in mind that the returned object MUST be detached,
      because this object can be written to another url for copying.


Tests
=====

:py:class:`neo.rawio.tests.common_rawio_test.BaseTestRawIO` and :py:class:`neo.test.io.commun_io_test.BaseTestIO` provide standard tests.
To use these you need to upload some sample data files at `gin-gnode`_. They will be publicly accessible for testing Neo.
These tests:

  * check the compliance with the schema: hierachy, attribute types, ...
  * For IO modules able to both write and read data, it compares a generated dataset with the same data after a write/read cycle.

The test scripts download all files from `gin-gnode`_ and stores them locally in ``/tmp/files_for_tests/``.
Subsequent test runs use the previously downloaded files, rather than trying to download them each time.

Each test must have at least one class that inherits ``BaseTestRawIO`` and that has 3 attributes:
  * ``rawioclass``: the class
  * ``entities_to_test``: a list of files (or directories) to be tested one by one
  * ``files_to_download``: a list of files to download (sometimes bigger than ``entities_to_test``)

Here is an example test script taken from the distribution: :file:`test_axonrawio.py`:

.. literalinclude:: ../../neo/rawio/tests/test_axonrawio.py


Logging
=======

All IO classes by default have logging using the standard :mod:`logging` module: already set up.
The logger name is the same as the fully qualified class name, e.g. :class:`neo.io.hdf5io.NeoHdf5IO`.
The :attr:`class.logger` attribute holds the logger for easy access.

There are generally 3 types of situations in which an IO class should use a logger

  * Recoverable errors with the file that the users need to be notified about.
    In this case, please use :meth:`logger.warning` or :meth:`logger.error`.
    If there is an exception associated with the issue, you can use :meth:`logger.exception` in the exception handler to automatically include a backtrace with the log.
    By default, all users will see messages at this level, so please restrict it only to problems the user absolutely needs to know about.
  * Informational messages that advanced users might want to see in order to get some insight into the file.
    In this case, please use :meth:`logger.info`.
  * Messages useful to developers to fix problems with the io class.
    In this case, please use :meth:`logger.debug`.

A log handler is automatically added to :mod:`neo`, so please do not user your own handler.
Please use the :attr:`class.logger` attribute for accessing the logger inside the class rather than :meth:`logging.getLogger`.
Please do not log directly to the root logger (e.g. :meth:`logging.warning`), use the class's logger instead (:meth:`class.logger.warning`).
In the tests for the io class, if you intentionally test broken files, please disable logs by setting the logging level to `100`.


ExampleIO
=========

.. autoclass:: neo.rawio.ExampleRawIO

.. autoclass:: neo.io.ExampleIO

Here is the entire file:

.. literalinclude:: ../../neo/rawio/examplerawio.py

.. literalinclude:: ../../neo/io/exampleio.py


.. _`mailing list`: http://groups.google.com/group/neuralensemble
.. _gin-gnode: https://web.gin.g-node.org/NeuralEnsemble/ephy_testing_data
