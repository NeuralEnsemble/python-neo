****************
Neo Examples
****************

.. currentmodule:: neo

Introduction
=============

A list of examples in neo/examples/ illustrate how using neo classes.



NeuroConvert
===============

NeuroConvert is simple GUI written in PyQt4 and using neo.io.
You can browse the code it is quite intuitive.

NeuroConvert provide a convertion from/to format that offer Block or Segment at top hierachy level.
It is the case for almost formats.

To use it : python NeuroConvert.py in its place.
2 possible actions :
 - add a file to be converted in the list
 - start convertion
 
When insert a file (or a list of files with identic options):
 - choose input and output format
 - the list of file to convert
 - for some format you need to detail some reading options
 - for some format you need to detail some writing options
 - check for global convertion options

The list of input and output format is generated from the neo.io module. The list is bound to grow with the number of formats implementation.
This list is platform dependend. For instance, on win32 platform neo.io implement a binding of neuroshare, since neuroshare is dll based this bindings
will only works on this platform.



