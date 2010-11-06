Read the documentation.

To build the documentaion type this ina console:

sudo apt-get python-spinx subversion
svn https://neuralensemble.org/svn/neo
cd neo/trunk/doc
make html
firefox build/html/index.html
