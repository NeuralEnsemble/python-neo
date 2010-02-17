***********
Neo IO
***********

.. currentmodule:: neo

Introduction
===============

neo with the set of classes propose a simple and flexible IO API that fit many formats specifications.
neo.io is not only file oriented it can also read/write objects from database.

Detail of API
================

The neo.io API is simple :
 - each format has its IO classes for examples for Spike2 files you have a Spike2IO class
 - each class inherit BaseIO class
 - each io can read or write one or several classes of neo, for example Segment, SpikeTrainList, Block, ...
 - each io declare in its class attributes what it is able to read or write.


.. autoclass:: neo.io.baseio.BaseIO



List of implemented formats
=================================

.. automodule::neo.io.asciisignal




Impletation of a new IO
===========================

ExampleIO is a fake IO just for illustrating how to implement a IO.


.. autoclass:: neo.io.ExampleIO

