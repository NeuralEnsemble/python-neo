.. _specific_annotations:

********************
Specific annotations
********************

Introduction
------------

Neo imposes and recommends some attributes for all objects, and also provides
the *annotations* dict for all objects to deal with any kind of extensions.
This flexible feature allow Neo objects to be customized for many use cases.

While any names can be used for annotations, interoperability will be improved
if there is some consistency in naming. Here we suggest some conventions for
annotation names.


Patch clamp
-----------

.. todo: TODO


Network simultaion
------------------


Spike sorting
-------------

**SpikeTrain.annotations['waveform_features']** : when spike sorting the
waveform is reduced to a smaller dimensional space with PCA or wavelets. This
attribute is the projected matrice. NxM (N spike number, M features number.
KlustakwikIO supports this feature.











