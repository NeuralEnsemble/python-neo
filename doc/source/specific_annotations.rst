.. _specific_annotations:

********************
Specific annotations
********************

Introduction
------------

Neo imposes and recommends some attributes for all objects.
But neo also introduce the *annotations* dict for all objects to deal with any kind 
of extentions.
This flexible features allow to custumize neo objects for many usecases.
Here is a list field by field for conventionnal attributes that could be used.


Patch clamp
-----------



Network simultaion
------------------


Spike sorting
-------------

**SpikeTrain.annotations['waveform_features']** : when spike sort the waveform is reduced to a smaller dimentional space with PCA or wavelet. This attributes is
the projected matrice. NxM (N spike number, M features number. KlustakwikIO supports this features.











