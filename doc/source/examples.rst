****************
Neo Examples
****************

.. currentmodule:: neo

Introduction
=============

A set of examples in neo/examples/ illustrates the use of neo classes.



NeuroConvert
===============

NeuroConvert is a simple GUI written in PyQt4 and using neo.io.
You can browse the code: it is quite intuitive.

NeuroConvert provide a conversion from/to formats that offer Block or Segment at the top level of the hierarchy,
which is the case for almost all supported formats.

To use it::

    python NeuroConvert.py
    
in the examples/NeuroConvert/ directory

There are two possible actions:
 - add a file to be converted to the list
 - start conversion
 
When inserting a file (or a list of files with identical options):
 - choose input and output format
 - the list of file to convert
 - for some formats you need to detail some reading options
 - for some formats you need to detail some writing options
 - check for global conversion options

The list of input and output formats is generated from the neo.io module. The list is bound to grow with the number of formats implemented.
This list is platform dependent. For instance, on the win32 platform neo.io implements a binding of neuroshare (since neuroshare is dll-based this binding
will only work on this platform).



