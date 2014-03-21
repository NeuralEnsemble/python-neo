# -*- coding: utf-8 -*-
"""
Tests of the neo.core.container.Container class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from neo.core.container import Container, unique_objs


class Test_unique_objs(unittest.TestCase):
    '''
    TestCase for unique_objs
    '''
    def test_some(self):
        a = 1
        b = np.array([3.14159265, 3.1415])
        c = [1, '1', 2.3, '5 8']
        d = set([1, '2', 'spam'])

        objs = [a, b, b, b, c, b, a, d, b, b, a, d, d, d, c, d, b, d, c, a]
        targ = [a, b, c, d]
        res = unique_objs(objs)
        self.assertEqual(targ, res)


class TestContainerNeo(unittest.TestCase):
    '''
    TestCase to make sure basic initialization and methods work
    '''
    def test_init(self):
        '''test to make sure initialization works properly'''
        container = Container(name='a container', description='this is a test')
        self.assertEqual(container.name, 'a container')
        self.assertEqual(container.description, 'this is a test')
        self.assertEqual(container.file_origin, None)

    def test__children(self):
        container = Container()
        self.assertEqual(container._single_parent_objects, ())
        self.assertEqual(container._multi_parent_objects, ())

        self.assertEqual(container._single_parent_containers, ())
        self.assertEqual(container._multi_parent_containers, ())

        self.assertEqual(container._parent_objects, ())
        self.assertEqual(container._parent_containers, ())

        self.assertEqual(container._container_child_objects, ())
        self.assertEqual(container._data_child_objects, ())
        self.assertEqual(container._multi_child_objects, ())
        self.assertEqual(container._child_properties, ())

        self.assertEqual(container._repr_pretty_containers, ())

        self.assertEqual(container._single_child_objects, ())

        self.assertEqual(container._container_child_containers, ())
        self.assertEqual(container._data_child_containers, ())
        self.assertEqual(container._single_child_containers, ())
        self.assertEqual(container._multi_child_containers, ())

        self.assertEqual(container._child_objects, ())
        self.assertEqual(container._child_containers, ())

        self.assertEqual(container._multi_children, ())
        self.assertEqual(container._single_children, ())
        self.assertEqual(container.data_children, ())
        self.assertEqual(container.container_children, ())
        self.assertEqual(container.children, ())
        self.assertEqual(container.parents, ())

        self.assertEqual(container.data_children_recur, ())
        self.assertEqual(container.container_children_recur, ())
        self.assertEqual(container.children_recur, ())

        self.assertEqual(container.filter(test=1), [])
        self.assertEqual(container.filter(data=True, container=False, test=1),
                         [])
        self.assertEqual(container.filter(data=False, container=False, test=1),
                         [])
        self.assertEqual(container.filter(data=True, container=True, test=1),
                         [])
        self.assertEqual(container.filter(data=False, container=True, test=1),
                         [])

        self.assertEqual(container.size, {})

        container.create_many_to_one_relationship()
        container.create_many_to_many_relationship()
        container.create_relationship()


class Test_Container_merge(unittest.TestCase):
    '''
    TestCase to make sure merge method works
    '''
    def setUp(self):
        self.name1 = 'a container 1'
        self.name2 = 'a container 2'
        self.description1 = 'this is a test 1'
        self.description2 = 'this is a test 2'
        self.cont1 = Container(name=self.name1, description=self.description1)
        self.cont2 = Container(name=self.name2, description=self.description2)

    def test_merge__dict(self):
        self.cont1.annotations = {'val1': 1, 'val2': 2.2, 'val3': 'test1'}
        self.cont2.annotations = {'val2': 2.2, 'val3': 'test2',
                                  'val4': [4, 4.4], 'val5': True}

        ann1 = self.cont1.annotations
        ann1c = self.cont1.annotations.copy()
        ann2c = self.cont2.annotations.copy()

        targ = {'val1': 1, 'val2': 2.2, 'val3': 'test1;test2',
                'val4': [4, 4.4], 'val5': True}

        self.cont1.merge(self.cont2)

        self.assertEqual(ann1, self.cont1.annotations)
        self.assertNotEqual(ann1c, self.cont1.annotations)
        self.assertEqual(ann2c, self.cont2.annotations)
        self.assertEqual(targ, self.cont1.annotations)

        self.assertEqual(self.name1, self.cont1.name)
        self.assertEqual(self.name2, self.cont2.name)
        self.assertEqual(self.description1, self.cont1.description)
        self.assertEqual(self.description2, self.cont2.description)

    def test_merge__different_type_AssertionError(self):
        self.cont1.annotations = {'val1': 1, 'val2': 2.2, 'val3': 'tester'}
        self.cont2.annotations = {'val3': False, 'val4': [4, 4.4],
                                  'val5': True}
        self.assertRaises(AssertionError, self.cont1.merge, self.cont2)

    def test_merge__unmergable_unequal_AssertionError(self):
        self.cont1.annotations = {'val1': 1, 'val2': 2.2, 'val3': True}
        self.cont2.annotations = {'val3': False, 'val4': [4, 4.4],
                                  'val5': True}
        self.assertRaises(AssertionError, self.cont1.merge, self.cont2)


@unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
class Test_pprint(unittest.TestCase):
    def test__pretty(self):
        name = 'an object'
        description = 'this is a test'
        obj = Container(name=name, description=description)
        res = pretty(obj)
        targ = "Container with  name: '%s' description: '%s'" % (name,
                                                                 description)
        self.assertEqual(res, targ)


if __name__ == "__main__":
    unittest.main()
