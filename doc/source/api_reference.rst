======================
Neo core API Reference
======================

Relationships between Neo objects
=================================

Object:
  * With a star = inherits from :class:`Quantity`
Attributes:
  * In red = required
  * In white = recommended
Relationships:
  * In cyan = one to many


.. image:: images/simple_generated_diagram.png
    :width: 750 px

:download:`Click here for a better quality SVG diagram <./images/simple_generated_diagram.svg>`

.. note:: This figure does not include :class:`ChannelView` and :class:`RegionOfInterest`.



.. automodule:: neo.core

.. testsetup:: *

    from neo import SpikeTrain
    import quantities as pq
