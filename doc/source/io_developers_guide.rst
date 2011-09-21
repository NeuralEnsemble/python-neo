.. _io_dev_guide:

****************************
IO developers' guide
****************************


.. _io_guiline:

Guideline for IO implementation
===========================

Receipe to develop an IO for a new format:
    1. Fully understand the object model. See core_page_. If doubt ask the list.
    2. Fully understand neo.io.exampleio, It is a fake IO to explain the API. If doubt ask the list.
    3. Copy/paste examplio.py and choose a clear file and class names for your IO.
    4. Make the point on which **supported_objects** and **readable_objects** your IO will deal with. This is the crutial point.
    5. Implement all methods read_XXX related to **readable_objects**.
    6. Write good docstring. Precise dependencies.
    7. Add your class in neo.io.__init__ Keep the import inside try/except for depencies reasons.
    8. Contact authors to put sample files for testing on the G-Node server. The write acces is not public.
    9. Write  test in neo/test/io/test_xxxxxxxxio.py. You must at least pass the standard test (hinerited with BaseTestIO).
    10. Commit or send patch only if all tests are compliants.


Misclaneous
============================

Notes:
    * if your IO support several version of a format (like ABF1, ABF2) you must put all cases for your files
    * :py:func:`neo.io.tools.create_many_to_one_relationship` offer utility to complete hierachy when all on_to_many relationship are done.
    * :py:func:`neo.io.tools.populate_RecordingChannel` offer utility to create inside a Block all RecordingChannel and links to AnalogSignal, SpikeTrain, ...
    * Precise in docstring where did you get the file format specification if it is a closed one.
    * If your IO is based on database mapper. keep in mind that the returned object MUST be detached. Because this object can be used to be written in an other url for copying.
    

Tests
=============================

:py:class:`neo.test.io.commun_io_test.BaseTestIO` provide standard test.
For that you need to upload some files at G-Node. They will publicy accessible for testing neo.
Theses tests:
  * check the compliance with schema: hierachy, type of attributes, ...
  * check if IO respect *lazy* and *cascade* keyword.
  * For IO able to write and read. It compare a generated schema and the same after write/read cycle.

Theses test download locally all files from `g-node <https://portal.g-node.org/neo/>`_. It is done only once in neo/test/io/files_for_tests.
Theses files are not deleted to avoid many download for developpers and testers.
So if you lanch tests only once do not forget to delete this directory.


This is an example of test_axonio.py, note that the list of files is present at G-Node.

.. literalinclude:: ../../neo/test/io/test_axonio.py



ExampleIO
==============================

.. autoclass:: neo.io.ExampleIO

Here the file with all coments:

.. literalinclude:: ../../neo/io/exampleio.py



