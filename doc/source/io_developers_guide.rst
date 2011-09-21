.. _io_dev_guide:

****************************
IO developers' guide
****************************





Implementation of a new IO
===========================

ExampleIO is a fake IO just for illustrating how to implement a IO. Developers who would like to write their own file format
should take a look to better understand the structure. One can also refer to the :class:`BaseIO` generic file.

As was already said, the default read/write methods should return the highest object in the hierarchy, 
which is often a :class:`Block` or a :class:`Segment`. Individual read/write methods should be implemented as much as possible.

.. autoclass:: neo.io.ExampleIO

.. autoclass:: neo.io.baseio.BaseIO


For advice or comments on the coding guidelines, developers can send e-mail to sgarcia@olfac.univ-lyon1.fr.


Steps
===========================

Steps for developping:
  1 - 
  2 - 
  3 - 


.. DETACHED
