=================
Developers' guide
=================

These instructions are for developing on a Unix-like platform, e.g. Linux or
Mac OS X, with the bash shell. If you develop on Windows, please get in touch.


Mailing lists
-------------

General discussion of Neo development takes place in the `NeuralEnsemble Google
group`_.

Discussion of issues specific to a particular ticket in the issue tracker
should take place on the tracker.


Using the issue tracker
-----------------------

If you find a bug in Neo, please create a new ticket on the `issue tracker`_,
setting the type to "defect".
Choose a name that is as specific as possible to the problem you've found, and
in the description give as much information as you think is necessary to
recreate the problem. The best way to do this is to create the shortest
possible Python script that demonstrates the problem, and attach the file to
the ticket.

If you have an idea for an improvement to Neo, create a ticket with type
"enhancement". If you already have an implementation of the idea, create a
patch (see below) and attach it to the ticket.

To keep track of changes to the code and to tickets, you can register for
a GitHub account and then set to watch the repository at `GitHub Repository`_
(see https://help.github.com/articles/watching-repositories/).

Requirements
------------

    * Python_ 2.6, 2.7, or 3.3
    * numpy_ >= 1.3.0
    * quantities_ >= 0.9.0
    * if using Python 2.6, unittest2_ >= 0.5.1
    * Setuptools >= 0.7
    * nose_ >= 0.11.1 (for running tests)
    * Sphinx_ >= 0.6.4 (for building documentation)
    * (optional) tox_ >= 0.9 (makes it easier to test with multiple Python versions)
    * (optional) coverage_ >= 2.85 (for measuring test coverage)
    * (optional) scipy >= 0.8 (for Matlab IO)
    * (optional) pytables >= >= 2.2 (for HDF5 IO)


Getting the source code
-----------------------

We use the Git version control system. The best way to contribute is through
GitHub_. You will first need a GitHub account, and you should then fork the
repository at `GitHub Repository`_
(see http://help.github.com/fork-a-repo/).

To get a local copy of the repository::

    $ cd /some/directory
    $ git clone git@github.com:<username>/python-neo.git
    
Now you need to make sure that the ``neo`` package is on your PYTHONPATH.
You can do this either by installing Neo::

    $ cd python-neo
    $ python setup.py install
    $ python3 setup.py install

(if you do this, you will have to re-run ``setup.py install`` any time you make
changes to the code) *or* by creating symbolic links from somewhere on your
PYTHONPATH, for example::

    $ ln -s python-neo/neo
    $ export PYTHONPATH=/some/directory:${PYTHONPATH}

An alternate solution is to install Neo with the *develop* option, this avoids
reinstalling when there are changes in the code::

    $ sudo python setup.py develop

To update to the latest version from the repository::

    $ git pull


Running the test suite
----------------------

Before you make any changes, run the test suite to make sure all the tests pass
on your system::

    $ cd neo/test

With Python 2.7 or 3.3::

    $ python -m unittest discover
    $ python3 -m unittest discover

If you have nose installed::

    $ nosetests

At the end, if you see "OK", then all the tests
passed (or were skipped because certain dependencies are not installed),
otherwise it will report on tests that failed or produced errors.

To run tests from an individual file::

    $ python test_analogsignal.py
    $ python3 test_analogsignal.py


Writing tests
-------------

You should try to write automated tests for any new code that you add. If you
have found a bug and want to fix it, first write a test that isolates the bug
(and that therefore fails with the existing codebase). Then apply your fix and
check that the test now passes.

To see how well the tests cover the code base, run::

    $ nosetests --with-coverage --cover-package=neo --cover-erase


Working on the documentation
----------------------------

All modules, classes, functions, and methods (including private and subclassed
builtin methods) should have docstrings.
Please see `PEP257`_ for a description of docstring conventions.

Module docstrings should explain briefly what functions or classes are present.
Detailed descriptions can be left for the docstrings of the respective
functions or classes.  Private functions do not need to be explained here.

Class docstrings should include an explanation of the purpose of the class
and, when applicable, how it relates to standard neuroscientific data.
They should also include at least one example, which should be written
so it can be run as-is from a clean newly-started Python interactive session
(that means all imports should be included).  Finally, they should include
a list of all arguments, attributes, and properties, with explanations.
Properties that  return data calculated from other data should explain what
calculation is done.  A list of methods is not needed, since documentation
will be generated from the method docstrings.

Method and function docstrings should include an explanation for what the
method or function does.  If this may not be clear, one or more examples may
be included.  Examples that are only a few lines do not need to include
imports or setup, but more complicated examples should have them.

Examples can be tested easily using th iPython %doctest_mode magic.  This will
strip >>> and ... from the beginning of each line of the example, so the
example can be copied and pasted as-is.

The documentation is written in `reStructuredText`_, using the `Sphinx`_
documentation system. Any mention of another neo module, class, attribute,
method, or function should be properly marked up so automatic
links can be generated.  The same goes for quantities or numpy.

To build the documentation::

    $ cd python-neo/doc
    $ make html

Then open `some/directory/neo_trunk/doc/build/html/index.html` in your browser.

Committing your changes
-----------------------

Once you are happy with your changes, **run the test suite again to check
that you have not introduced any new bugs**. It is also recommended to check
your code with a code checking program, such as `pyflakes`_ or `flake8`_.  Then
you can commit them to your local repository::

    $ git commit -m 'informative commit message'

If this is your first commit to the project, please add your name and
affiliation/employer to :file:`doc/source/authors.rst`

You can then push your changes to your online repository on GitHub::

    $ git push

Once you think your changes are ready to be included in the main Neo repository,
open a pull request on GitHub
(see https://help.github.com/articles/using-pull-requests).


Python 3
--------

Neo core should work with both recent versions of Python 2 (versions 2.6 and
2.7) and Python 3 (version 3.3). Neo IO modules should ideally work with both
Python 2 and 3, but certain modules may only work with one or the other
(see :doc:`install`).

So far, we have managed to write code that works with both Python 2 and 3.
Mainly this involves avoiding the ``print`` statement (use ``logging.info``
instead), and putting ``from __future__ import division`` at the beginning of
any file that uses division.

If in doubt, `Porting to Python 3`_ by Lennart Regebro is an excellent resource.

The most important thing to remember is to run tests with at least one version
of Python 2 and at least one version of Python 3. There is generally no problem
in having multiple versions of Python installed on your computer at once: e.g.,
on Ubuntu Python 2 is available as `python` and Python 3 as `python3`, while
on Arch Linux Python 2 is `python2` and Python 3 `python`. See `PEP394`_ for
more on this.


Coding standards and style
--------------------------

All code should conform as much as possible to `PEP 8`_, and should run with
Python 2.6, 2.7, and 3.3.

You can use the `pep8`_ program to check the code for PEP 8 conformity.
You can also use `flake8`_, which combines pep8 and pyflakes.

However, the pep8 and flake8 programs does not check for all PEP 8 issues.
In particular, they does not check that the import statements are in the
correct order.

Also, please do not use `from xyz import *`.  This is slow, can lead to
conflicts, and makes it difficult for code analysis software.


Making a release
----------------

.. TODO: discuss branching/tagging policy.

Add a section in /doc/src/whatisnew.rst for the release.

First check that the version string (in :file:`neo/version.py`,
:file:`setup.py`, :file:`doc/conf.py` and :file:`doc/install.rst`) is correct.

To build a source package::

    $ python setup.py sdist

To upload the package to `PyPI`_ (currently Samuel Garcia and Andrew Davison
have the necessary permissions to do this)::

    $ python setup.py sdist upload
    $ python setup.py upload_docs --upload-dir=doc/build/html

.. should we also distribute via software.incf.org

Finally, tag the release in the Git repository and push it::

    $ git tag <version>
    $ git push --tags origin
    

.. make a release branch


If you want to develop your own IO module
-----------------------------------------

See :ref:`io_dev_guide` for implementation of a new IO.




.. _Python: http://www.python.org
.. _nose: http://somethingaboutorange.com/mrl/projects/nose/
.. _unittest2: http://pypi.python.org/pypi/unittest2
.. _Setuptools: https://pypi.python.org/pypi/setuptools/
.. _tox: http://codespeak.net/tox/
.. _coverage: http://nedbatchelder.com/code/coverage/
.. _`PEP 8`: http://www.python.org/dev/peps/pep-0008/
.. _`issue tracker`: https://github.com/NeuralEnsemble/python-neo/issues
.. _`Porting to Python 3`: http://python3porting.com/
.. _`NeuralEnsemble Google group`: http://groups.google.com/group/neuralensemble
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _Sphinx: http://sphinx.pocoo.org/
.. _numpy: http://numpy.scipy.org/
.. _quantities: http://pypi.python.org/pypi/quantities
.. _PEP257: http://www.python.org/dev/peps/pep-0257/
.. _PEP394: http://www.python.org/dev/peps/pep-0394/
.. _PyPI: http://pypi.python.org
.. _GitHub: http://github.com
.. _`GitHub Repository`: https://github.com/NeuralEnsemble/python-neo/
.. _pep8: https://pypi.python.org/pypi/pep8
.. _flake8: https://pypi.python.org/pypi/flake8/
.. _pyflakes: https://pypi.python.org/pypi/pyflakes/
