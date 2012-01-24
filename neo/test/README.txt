To run all tests:

With Python 2.7 or 3.2:

    $ python -m unittest discover

If you have nose installed:

    $ nosetests


To run tests from an individual file:

    $ python test_analogsignal.py
    
(in all Python versions)



For coverage
nosetests --with-coverage --cover-erase --cover-package=neo