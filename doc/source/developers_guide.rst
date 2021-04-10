=================
Developers' guide
=================

These instructions are for developing on a Unix-like platform, e.g. Linux or
macOS, with the bash shell. If you develop on Windows, please get in touch.


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
(see https://help.github.com/en/articles/watching-and-unwatching-repositories).

Requirements
------------

    * Python_ 3.5 or later
    * numpy_ >= 1.11.0
    * quantities_ >= 0.12.1
    * nose_ >= 1.1.2 (for running tests)
    * Sphinx_ (for building documentation)
    * (optional) coverage_ >= 2.85 (for measuring test coverage)
    * (optional) scipy >= 0.12 (for MatlabIO)
    * (optional) h5py >= 2.5 (for KwikIO)
    * (optional) nixio (for NixIO)
    * (optional) pillow (for TiffIO)

We strongly recommend you develop within a virtual environment (from virtualenv, venv or conda).

Getting the source code
-----------------------

We use the Git version control system. The best way to contribute is through
GitHub_. You will first need a GitHub account, and you should then fork the
repository at `GitHub Repository`_
(see http://help.github.com/en/articles/fork-a-repo).

To get a local copy of the repository::

    $ cd /some/directory
    $ git clone git@github.com:<username>/python-neo.git

Now you need to make sure that the ``neo`` package is on your PYTHONPATH.
You can do this either by installing Neo::

    $ cd python-neo
    $ python3 setup.py install

(if you do this, you will have to re-run ``setup.py install`` any time you make
changes to the code) *or* by creating symbolic links from somewhere on your
PYTHONPATH, for example::

    $ ln -s python-neo/neo
    $ export PYTHONPATH=/some/directory:${PYTHONPATH}

An alternate solution is to install Neo with the *develop* option, this avoids
reinstalling when there are changes in the code::

    $ sudo python setup.py develop

or using the "-e" option to pip::

    $ pip install -e python-neo

To update to the latest version from the repository::

    $ git pull


Running the test suite
----------------------

Before you make any changes, run the test suite to make sure all the tests pass
on your system::

    $ cd neo/test
    $ python3 -m unittest discover

If you have nose installed::

    $ nosetests

At the end, if you see "OK", then all the tests
passed (or were skipped because certain dependencies are not installed),
otherwise it will report on tests that failed or produced errors.

To run tests from an individual file::

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

Examples can be tested easily using the iPython `%doctest_mode` magic.  This will
strip >>> and ... from the beginning of each line of the example, so the
example can be copied and pasted as-is.

The documentation is written in `reStructuredText`_, using the `Sphinx`_
documentation system. Any mention of another Neo module, class, attribute,
method, or function should be properly marked up so automatic
links can be generated.  The same goes for quantities or numpy.

To build the documentation::

    $ cd python-neo/doc
    $ make html

Then open `some/directory/python-neo/doc/build/html/index.html` in your browser.

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
(see https://help.github.com/en/articles/about-pull-requests).


Python version
--------------

Neo should work with Python 3.5 or newer. If you need support for Python 2.7,
use Neo v0.8.0 or earlier.


Coding standards and style
--------------------------

All code should conform as much as possible to `PEP 8`_, and should run with
Python 3.5 or newer.

You can use the `pep8`_ program to check the code for PEP 8 conformity.
You can also use `flake8`_, which combines pep8 and pyflakes.

However, the pep8 and flake8 programs do not check for all PEP 8 issues.
In particular, they do not check that the import statements are in the
correct order.

Also, please do not use ``from xyz import *``.  This is slow, can lead to
conflicts, and makes it difficult for code analysis software.


Making a release
----------------

.. TODO: discuss branching/tagging policy.

Add a section in :file:`/doc/source/whatisnew.rst` for the release.

First check that the version string (in :file:`neo/version.py`) is correct.

To build a source package::

    $ python setup.py sdist


Tag the release in the Git repository and push it::

    $ git tag <version>
    $ git push --tags origin
    $ git push --tags upstream


To upload the package to `PyPI`_ (currently Samuel Garcia,  Andrew Davison,
Michael Denker and Julia Sprenger have the necessary permissions to do this)::

    $ twine upload dist/neo-0.X.Y.tar.gz

.. talk about readthedocs



.. make a release branch


If you want to develop your own IO module
-----------------------------------------

See :ref:`io_dev_guide` for implementation of a new IO.




.. _Python: https://www.python.org
.. _nose: https://nose.readthedocs.io/
.. _Setuptools: https://pypi.python.org/pypi/setuptools/
.. _tox: http://codespeak.net/tox/
.. _coverage: https://coverage.readthedocs.io/
.. _`PEP 8`: https://www.python.org/dev/peps/pep-0008/
.. _`issue tracker`: https://github.com/NeuralEnsemble/python-neo/issues
.. _`Porting to Python 3`: http://python3porting.com/
.. _`NeuralEnsemble Google group`: https://groups.google.com/forum/#!forum/neuralensemble
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _Sphinx: http://www.sphinx-doc.org/
.. _numpy: https://numpy.org/
.. _quantities: https://pypi.org/project/quantities/
.. _PEP257: https://www.python.org/dev/peps/pep-0257/
.. _PEP394: https://www.python.org/dev/peps/pep-0394/
.. _PyPI: https://pypi.org
.. _GitHub: https://github.com
.. _`GitHub Repository`: https://github.com/NeuralEnsemble/python-neo/
.. _pep8: https://pypi.org/project/pep8/
.. _flake8: https://pypi.org/project/flake8/
.. _pyflakes: https://pypi.org/project/pyflakes/
