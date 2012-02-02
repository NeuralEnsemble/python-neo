=================
Developers' guide
=================

These instructions are for developing on a Unix-like platform, e.g. Linux or
Mac OS X, with the bash shell. If you develop on Windows, please get in touch.


Mailing lists
-------------

General discussion of Neo development takes place in the `NeuralEnsemble Google
group`_.

Discussion of issues specific to a particular ticket in the issue tracker should
take place on the tracker.

.. since the NeuralEnsemble group is private (requires login to read), perhaps
.. we should create an open group neo_users (anyone can read, login to post) as well?

Using the issue tracker
-----------------------

If you find a bug in Neo, please create a new ticket on the `issue tracker`_,
setting the type to "defect".
Choose a name that is as specific as possible to the problem you've found, and
in the description give as much information as you think is necessary to
recreate the problem. The best way to do this is to create the shortest possible
Python script that demonstrates the problem, and attach the file to the ticket.

If you have an idea for an improvement to Neo, create a ticket with type
"enhancement". If you already have an implementation of the idea, create a patch
(see below) and attach it to the ticket.

To keep track of changes to the code and to tickets, you can follow the
`RSS feed`_.

Requirements
------------

    * Python_ 2.6, 2.7, 3.1 or 3.2
    * numpy_ >= 1.3.0
    * quantities_ >= 0.9.0
    * nose_ >= 0.11.1
    * if using Python 2.6 or 3.1, unittest2_ >= 0.5.1
    * Distribute_ >= 0.6
    * Sphinx_ >= 0.6.4
    * (optional) tox_ >= 0.9 (makes it easier to test with multiple Python versions)
    * (optional) coverage_ >= 2.85 (for measuring test coverage)


Getting the source code
-----------------------

We use the Subversion version control system. To get a copy of the latest
development version::

    $ cd /some/directory
    $ svn checkout https://neuralensemble.org/svn/neo/trunk neo_trunk
    
Now you need to make sure that the ``neo`` package is on your PYTHONPATH.
You can do this either by installing Neo::

    $ cd neo_trunk
    $ python setup.py install
    $ python3 setup.py install

(if you do this, you will have to re-run ``setup.py install`` any time you make
changes to the code) *or* by creating symbolic links from somewhere on your
PYTHONPATH, for example::

    $ ln -s neo_trunk/neo
    $ export PYTHONPATH=/some/directory:${PYTHONPATH}

An alternate solution is to install Neo with the *develop* option, this avoids
reinstalling when there are changes in the code::

    $ sudo python setup.py develop

To update to the latest version from the repository::

    $ svn update


Running the test suite
----------------------

Before you make any changes, run the test suite to make sure all the tests pass
on your system::

    $ cd neo/test

With Python 2.7 or 3.2::

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

The documentation is written in `reStructuredText`_, using the `Sphinx`_
documentation system. To build the documentation::

    $ cd neo_trunk/doc
    $ make html
    
Then open `some/directory/neo_trunk/doc/build/html/index.html` in your browser.

Committing your changes
-----------------------

Once you are happy with your changes, **run the test suite again to check
that you have not introduced any new bugs**. Then you can commit them to the
central repository, provided you have a NeuralEnsemble account::

    $ svn commit -m 'informative commit message'
    
If this is your first commit to the project, please add your name and
affiliation/employer to AUTHORS.txt.

If you do not have a NeuralEnsemble account, you can create a patch::

    $ svn diff > descriptive_name.patch
    
and attach it to a ticket in the `issue tracker`_. If you have made more than
one commit, determine the revision number of when you checked out or last updated
from the central repository (using ``svn log``), and then give a range of
revisions to include in the patch::

    $ svn diff start-revision:HEAD > descriptive_name.diff

To apply a patch to the source tree::

    $ patch -p0 < /path/to/site.descriptive_name.diff


Python 3
--------

Neo core should work with both recent versions of Python 2 (versions 2.6 and 2.7)
and Python 3. Neo IO modules should ideally work with both Python 2 and 3, but
certain modules may only work with one or the other (see :doc:install).

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
Python 2.6, 2.7, 3.1 and 3.2.


Making a release
----------------

.. TODO: discuss branching/tagging policy. We should really be developing in trunk

.. so doing a release also involves some svn copy commands and changing version number strings

First check that the version string (in ``neo/version.py``) is correct.

To build a source package::

    $ python setup.py sdist

To upload the package to `PyPI`_ (currently Samuel Garcia and Andrew Davison
have the necessary permissions to do this)::

    $ python setup.py sdist upload
    $ python setup.py upload_docs

.. I HAVEN'T TESTED THE upload_docs COMMAND YET

.. should we also distribute via software.incf.org

Finally, tag the release in the Subversion repository::

    $ svn cp https://neuralensemble.org/svn/neo/trunk https://neuralensemble.org/svn/neo/tags/<version>


If you want to develop your own IO module
-----------------------------------------

See :ref:`io_dev_guide` for implementation of a new IO.




.. _Python: http://www.python.org
.. _nose: http://somethingaboutorange.com/mrl/projects/nose/
.. _unittest2: http://pypi.python.org/pypi/unittest2
.. _Distribute: http://pypi.python.org/pypi/distribute
.. _tox: http://codespeak.net/tox/
.. _coverage: http://nedbatchelder.com/code/coverage/
.. _`PEP 8`: http://www.python.org/dev/peps/pep-0008/
.. _`issue tracker`: http://neuralensemble.org/trac/neo
.. _`Porting to Python 3`: http://python3porting.com/
.. _`NeuralEnsemble Google group`: http://groups.google.com/group/neuralensemble
.. _`RSS feed`: https://neuralensemble.org/trac/neo/timeline?changeset=on&milestone=on&ticket=on&wiki=on&max=50&daysback=90&format=rss
.. _`reStructuredText`: http://docutils.sourceforge.net/rst.html
.. _`Sphinx`: http://sphinx.pocoo.org/
.. _`numpy`: http://numpy.scipy.org/
.. _`quantities`: http://pypi.python.org/pypi/quantities
.. _`PEP394`: http://www.python.org/dev/peps/pep-0394/
.. _`PyPI`: http://pypi.python.org