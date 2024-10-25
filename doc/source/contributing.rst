===============================
Contributing to Neo development
===============================

Contributions to Neo, whether it be fixing bugs, adding support for new file formats, improving the documentation,
or adding new features, are welcome.
Many :doc:`contributors to Neo<authors>` are PhD students or postdocs, often with little or no previous experience of software development
or open-source projects, so as a community we try to be particularly welcoming to new contributors.
If you're new to open source software, `How to Contribute to Open Source`_ is a good introductory guide.


Discussing with other Neo developers
====================================

If you have an idea for improving Neo, the first step is to discuss it with other Neo developers
on the NeuralEnsemble `mailing list`_.

Discussion of issues specific to a particular ticket in the `issue tracker`_
should take place on the tracker.

.. note:: Note that the mailing list is shared with other projects in the general domain of computational and systems neuroscience,
          so please don't be confused if you see posts that don't relate to Neo!

.. goodforbeginners label

Setting up a development environment
====================================

These instructions are written for beginning developers.
If you are an experienced developer, you can skim this section.
If any steps are unclear or confusing, please let us know.


Fork the project on Github
--------------------------

To work on Neo you will need a Github account.
Once you have an account, go to the Neo project at https://github.com/NeuralEnsemble/python-neo
and click on the "Fork button" near the top-right corner (`more information on forking`_).

Now you can go to your own copy of the project at ``https://github.com/<your_username>/python-neo/``

Get the source code
-------------------

We use the Git version control software to work on Neo. Git makes it easier to keep track of changes to the Neo code.

If you don't have git installed on your computer, and/or haven't used git before, see the `GitHub Docs on git`_.

On the homepage of your own copy of Neo, click on the green "Code" button, and copy either the HTTPS or SSH link,
which will look like ``https://github.com/<your_username>/python-neo.git`` or ``git@github.com:<your_username>/python-neo.git``.
If you choose the HTTPS method you will be asked for your Github username and password regularly.
To avoid this, you can `add an SSH key to your Github account`_.

Now choose a location on your computer where you'd like to put the Neo source code, change to that directory,
and run *one* of the following commands (depending on whether you chose HTTPS or SSH):

::

    git clone https://github.com/<your_username>/python-neo.git

::

    git clone git@github.com:<your_username>/python-neo.git

This copies the "master" branch of the Neo code to a directory named "python-neo".
Now we need to make sure you can easily get the latest changes to Neo. Run::

    cd python-neo
    git remote -v

You should see something like::

    origin	git@github.com:<your_username>/python-neo.git (fetch)
    origin	git@github.com:<your_username>/python-neo.git (push)

which means that Git is using the alias "origin" to refer to your fork of the code.
So that you can easily download any ongoing changes to Neo after the initial "git clone",
we are going to add an alias "upstream" to refer to the main Neo repository::

    git remote add upstream https://github.com/NeuralEnsemble/python-neo.git


Create a virtual environment
----------------------------

It is strongly recommended to always work in a Python "virtual environment",
to avoid conflicts between different versions of packages you might need for different projects,
and to avoid causing problems with your operating system.
With a virtual environment, you are working in an isolated space, which contains
only the packages you specifically install into that environment.
You can, for example, create different environments with different versions of Python,
of NumPy, etc., and easily switch between them.

We give instructions here for :mod:`conda` and for Python's built-in :mod:`venv` tool.
Other virtual environment tools, such as virtualenv, are also available.
:mod:`venv` is a built-in Python module, so if you already have Python installed on your computer,
you can immediately get started without needing to install conda.
The disadvantage is that you can only use the same version(s) of Python as already installed on your system,
whereas conda allows you to easily create environments with different Python versions.

.. tab:: conda

    We suggest using Miniconda_, which provides a minimal Python environment,
    but still allows you to easily install packages from the Anaconda repository.

    Install Miniconda using the instructions on its website, then create a conda environment using::

        conda create --name my_neo_env

    where "my_neo_env" is a name for your environment: you can choose any name you wish.

    You then need to activate the environment::

        conda activate my_neo_env

    If you wish to leave the environment::

        conda deactivate


.. tab:: venv

    Unlike with :mod:`conda`, which stores your environment in a standard location,
    with :mod:`venv` you have to choose where to store the environment files.
    One option is to create a folder :file:`venv` in your home directory, and then
    create your virtual envs in subdirectories like :file:`venv/neo`.
    Another option is to create :file:`env` within the Neo source code folder, e.g.:

    .. tab:: Unix/macOS

        .. code-block:: bash

            python3 -m venv env

    .. tab:: Windows

        .. code-block:: bat

            py -m venv env

    You then need to activate the environment:

    .. tab:: Unix/macOS

        .. code-block:: bash

            source env/bin/activate

    .. tab:: Windows

        .. code-block:: bat

            .\env\Scripts\activate

    If you want to switch projects or otherwise leave your virtual environment, simply run::

        deactivate


Install dependencies
--------------------

The following commands will install all the packages needed to use Neo, run tests, and build the documentation.


.. tab:: conda

    .. code-block:: bash

        conda install --file requirements_dev.txt

        pip install -e .

.. tab:: venv

    The Neo testsuite uses Datalad_ to download test files. Datalad in turn
    depends on git-annex, which is not a Python package, and so cannot be installed
    with pip. See `installing Datalad`_ for instructions about installing git-annex
    on your system, then continue from here:

    .. code-block:: bash

        pip install -e .[test]

This does not install all the optional dependencies for different file formats.
If you're planning to work on a specific IO module, e.g. :class:`NixIO`,
you can install the dependencies for that module::

    pip install -e .[nixio]

Or you can install all optional dependencies with::

    pip install -e .[all]

.. note:: the "-e" flag is for "editable". It means that any changes you make to the Neo code will be immediately available
          in your virtual environment. Without this flag, you would need to re-run ``pip install`` every time you change the code.


Running the test suite
======================

To run the full test-suite, run::

    pytest

The first time this is run, all of the Neo test files will be downloaded to your machine,
so the run time can be an hour or more.
For subsequent runs, the files are already there, so the tests will run much faster.

Because Neo downloads datasets this can lead to issues in the course of offline development or
for packaging Neo (e.g. for a Linux distribution). In order to not download datasets and to skip
tests which require downloaded datasets the environment variable :code:`NEO_TESTS_NO_NETWORK` can
be set to any truthy value (e.g. :code:`'True'``).

For macOS/Linux this can be done by doing:

.. code-block:: bash

    NEO_TESTS_NO_NETWORK='True' pytest .

For Windows this can be done by doing:

.. code-block:: bat

    set NEO_TESTS_NO_NETWORK=true

    pytest . 

This can also be done with a conda environment variable if developing in a conda env. To configure these
see the docs at `conda env vars documentation`_.

It is often helpful to run only parts of the test suite. To test only the :mod:`neo.core` module,
which is much quicker than testing :mod:`neo.io`, run::

    pytest neo/test/coretest

You can also run only tests in a specific file, e.g.::

    pytest neo/test/coretest/test_analogsignal.py

and you can even run individual tests, e.g.::

    pytest neo/test/coretest/test_analogsignal.py::TestAnalogSignalConstructor::test__create_from_1d_quantities_array


Test coverage
-------------

A good way to start contributing to Neo is to improve the test coverage. If you run pytest with the "--cov" option:

::

    pytest --cov=neo --cov-report html --cov-report term

Then pytest will track which lines of Neo code are executed while running the tests,
and will generate a set of local web pages showing which lines have not been executed,
and hence not tested. Our goal is 100% coverage, so adding tests to increase coverage
is always welcome.


Writing tests
=============

You should try to write automated tests for any new code that you add. If you
have found a bug and want to fix it, first write a test that isolates the bug
(and that therefore fails with the existing codebase). Then apply your fix and
check that the test now passes.

We use Python's built-in :mod:`unittest` module to structure our tests,
which are placed in the :file:`neo/test` directory.
Related tests are grouped into classes, and each individual test is a method of that class,
so take a look at the existing tests before deciding whether to add a new method,
a new class or a new file.

Some guidelines on writing tests:

- Each test must be independent, the results shouldn't depend on the order in which tests are run.
- Fast tests are preferred: we have a lot of tests, so the total time adds up quickly.
- Where there are branches in the code (e.g. ``if`` statements), try to test all possible branches.


Coding standards and style
==========================

All code should conform as much as possible to `PEP 8`_, with a maximum line length of 99 characters,
and should run with Python 3.8 or newer.

You can use the `pep8`_ program to check the code for PEP 8 conformity.
You can also use `flake8`_, which combines pep8 and pyflakes.

However, the pep8 and flake8 programs do not check for all PEP 8 issues.
In particular, they do not check that the import statements are in the correct order
(standard library imports first, then other dependencies, then Neo's own code).


Working on the documentation
============================

All modules, classes, functions, and methods (including private and subclassed
builtin methods) should have docstrings.
Please see `PEP 257`_ for a description of docstring conventions.

Module docstrings should explain briefly what functions or classes are present.
Detailed descriptions can be left for the docstrings of the respective
functions or classes.

Class docstrings should include an explanation of the purpose of the class
and, when applicable, how it relates to standard neuroscience data.
They should also include at least one example, which should be written
so it can be run as-is from a clean, newly-started Python interactive session
(this means all imports should be included).  Finally, they should include
a list of all arguments, attributes, and properties, with explanations.
Properties that return data calculated from other data should explain what
calculation is done. A list of methods is not needed, since documentation
will be generated from the method docstrings.

Method and function docstrings should include an explanation of what the
method or function does. If this may not be clear, one or more examples may
be included. Examples that are only a few lines do not need to include
imports or setup, but more complicated examples should have them.

The documentation is written in `reStructuredText`_, using the `Sphinx`_
documentation system. Any mention of another Neo module, class, attribute,
method, or function should be properly marked up so automatic
links can be generated.

To build the documentation::

    pip install -e .[docs]
    cd doc
    make html

Then open :file:`build/html/index.html` in your browser.


Writing an IO module
====================

The topic of writing a new IO module is addressed in :doc:`add_file_format`.


Working with Git
================

If you're new to Git, there are many good learning resources on the web,
such as `Blischak, Davenport and Wilson (2016)`_.

We recommend the following best practices, based on `this document by Luis Matos`_:

**Commit related changes**
    Try to ensure that each commit contains only changes related to a single topic.
    For example, fixing two different bugs should result in two separate commits.
    Small commits make it easier for other developers to understand the changes.
    You don't have to commit all the changes in your working copy:
    use the `staging area`_ and the ability to stage only parts of a file to
    only include relevant changes.
    Graphical Git tools such as Sourcetree_ and GitKraken_ can make this very easy.

**Commit and push often**
    Committing often helps to (i) keep your commits small, (ii) commit only related changes,
    (iii) reduce the risk of losing work, and (iv) share your code more frequently with others,
    which is important for a fairly busy project like Neo because it makes it easier to integrate
    different people's changes.

**Test your code before you commit**
    Since the Neo IO module tests take a long time to run, you may not wish to run them before
    every commit. However, the core module tests run quickly, so you can run them every time,
    and of course if you're working on a particular IO module you should run the tests for that module.

**Write informative commit messages**
    The first line of your commit message should be a short summary of your changes.
    Then add a blank line. Then give additional information, for example explaining
    the motivation for the change and adding links to related Github issue(s).

**Use branches**
    **Always** work in a branch specific to the bug you're trying to fix or the feature
    you're trying to add, never in the master/main branch.
    Branches should ideally be short lived, to minimise the risk of conflicts when merging
    into the master branch.
    You should aim to synchronize the master/main branch in your own fork with the upstream
    NeuralEnsemble master branch frequently, and always create a new branch off this master
    branch.


Making a pull request
=====================

When you think your bug fix, new feature, or cleanup is ready to be merged into Neo,
`open a pull request on GitHub`_.

If this is your first pull request to the project,
please include a commit in which you add your name and affiliation/employer (if any)
to :file:`doc/source/authors.rst`.


Reviewing pull requests
=======================

In addition to writing code and documentation, constructive reviewing of other people's code
is also a great way to contribute to Neo development.

Please review the `Code of Conduct`_ before reviewing. Code reviews are a big part of what determines
whether contributing to Neo is a positive experience or not.

Like reviewing scientific papers, reviewing pull requests (PRs) requires both attention to detail
and seeing the big picture.

**Attention to detail**
    - For every line of code added or removed, ensure you understand the purpose of the change.
      If this is unclear, add a comment in the PR.
    - Ensure that unit tests have been added or modified appropriately to test the modified code.
    - Check that all of the automated checks have passed.
      Occasionally a test failure is not related to the specific pull request,
      but to some change in the CI system, or some other unrelated change.
      If you're *absolutely sure* this is the case, then it's ok to approve the PR,
      but please add a comment explaining why you think the failure is unrelated.
    - If you see a more efficient way to implement something, feel free to add a comment
      about this, but please ensure all comments are polite and constructive.
    - Check that new code is adequately documented, with docstrings for new functions or classes,
      comments to explain complex pieces of code, and changes to the user guide if adding or removing features.
    - Ensure there is no temporary, commented out code.
      Since Neo is a library, there should be no, or very few ``print()`` functions.

**The big picture**
    - Is this code maintainable? If the algorithm is highly complex so that few people can understand it,
      or it's a very niche feature, then it may not be possible to maintain this code in the long term.
    - Are you aware of other people working on the same issue, or a related one?
      If that's the case, please make the submitter aware of this so they can coordinate with the others.
    - Pull requests don't have to be perfect, especially if they're from first-time or
      inexperienced contributors. Sometimes it's ok to accept a partial or sub-optimal solution
      that can be improved later, as long as it moves the project in a good direction.


Making a release
================

- Create a new file with the release notes in the folder :file:`doc/source/releases`
  then add a link to it in :file:`/doc/source/releases.rst`.
- Ensure you are in the master branch.
- Check that all tests pass, and that the documentation builds correctly (see above).
- Tag the release in the Git repository and push it::

    git tag <version>
    git push --tags origin
    git push --tags upstream

- Wait for the `continuous integration server`_ to run all the tests, ensure there are
  no failures. If there are failures, fix them, and move the tag to the new commit.
- Build source and wheel packages::

    python -m build

- Upload the package to `PyPI`_ (the members of the :ref:`section-maintainers` team have the necessary permissions to do this)::

    twine upload dist/neo-0.X.Y.tar.gz

- Check the `Read the Docs`_ documentation has built correctly.
  If not, you may need to manually activate the new version in the `docs configuration page`_.


.. _`mailing list`: http://neuralensemble.org/community
.. _`issue tracker`: https://github.com/NeuralEnsemble/python-neo/issues/
.. _`add an SSH key to your Github account`: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
.. _`more information on forking`: http://help.github.com/en/articles/fork-a-repo
.. _`GitHub Docs on git`: https://docs.github.com/en/get-started/quickstart/set-up-git
.. _`How to Contribute to Open Source`: https://opensource.guide/how-to-contribute/
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Datalad: https://www.datalad.org
.. _`installing Datalad`: https://handbook.datalad.org/en/latest/intro/installation.html#installation-and-configuration
.. _pep8: https://pypi.org/project/pep8/
.. _flake8: https://pypi.org/project/flake8/
.. _pyflakes: https://pypi.org/project/pyflakes/
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _Sphinx: http://www.sphinx-doc.org/
.. _`PEP 257`: https://www.python.org/dev/peps/pep-0257/
.. _`PEP 8`: https://www.python.org/dev/peps/pep-0008/
.. _pep8: https://pypi.org/project/pep8/
.. _`Blischak, Davenport and Wilson (2016)`: https://doi.org/10.1371/journal.pcbi.1004668
.. _`this document by Luis Matos`: https://gist.github.com/luismts/
.. _`staging area`: https://coderefinery.github.io/git-intro/staging-area/
.. _Sourcetree: https://www.sourcetreeapp.com
.. _GitKraken: https://www.gitkraken.com
.. _`open a pull request on GitHub`: https://help.github.com/en/articles/about-pull-requests
.. _`Code of Conduct`: https://github.com/NeuralEnsemble/python-neo/blob/master/CODE_OF_CONDUCT.md
.. _`maintainers team`: https://github.com/orgs/NeuralEnsemble/teams/neo-maintainers
.. _PyPI: https://pypi.org/project/neo
.. _`continuous integration server`: https://github.com/NeuralEnsemble/python-neo/actions
.. _`Read the Docs`: https://neo.readthedocs.io/en/latest/
.. _`docs configuration page`: https://readthedocs.org/projects/neo/
.. _`conda env vars documentation`: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment