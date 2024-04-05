"""
Tests of the neo.core.container.Container class
"""

import unittest

import quantities as pq

import numpy as np

import neo.core
from neo.core import filters

try:
    from IPython.lib.pretty import pretty
except ImportError as err:
    HAVE_IPYTHON = False
else:
    HAVE_IPYTHON = True

from neo.core.container import Container, unique_objs


class Test_unique_objs(unittest.TestCase):
    """
    TestCase for unique_objs
    """

    def test_some(self):
        a = 1
        b = np.array([3.14159265, 3.1415])
        c = [1, "1", 2.3, "5 8"]
        d = {1, "2", "spam"}

        objs = [a, b, b, b, c, b, a, d, b, b, a, d, d, d, c, d, b, d, c, a]
        targ = [a, b, c, d]
        res = unique_objs(objs)
        self.assertEqual(targ, res)


class TestContainerNeo(unittest.TestCase):
    """
    TestCase to make sure basic initialization and methods work
    """

    @classmethod
    def setUpClass(cls):
        seg = neo.core.Segment()
        st1 = neo.core.SpikeTrain([1, 2] * pq.ms, t_stop=10)
        st1.annotate(test=5)
        st2 = neo.core.SpikeTrain([3, 4] * pq.ms, t_stop=10)
        st2.annotate(test=6)
        st2.annotate(name="st_num_1")
        st2.annotate(filt=6)
        st3 = neo.core.SpikeTrain([5, 6] * pq.ms, t_stop=10)
        st3.annotate(list=[1, 2])
        st3.annotate(dict={"key": 5})
        seg.spiketrains.append(st1)
        seg.spiketrains.append(st2)
        seg.spiketrains.append(st3)

        cls.seg = seg
        cls.st1 = st1
        cls.st2 = st2

    def test_init(self):
        """test to make sure initialization works properly"""
        container = Container(name="a container", description="this is a test")
        self.assertEqual(container.name, "a container")
        self.assertEqual(container.description, "this is a test")
        self.assertEqual(container.file_origin, None)

    def test__children(self):
        container = Container()
        self.assertEqual(container._parent_objects, ())

        self.assertEqual(container._parent_containers, ())

        self.assertEqual(container._parent_objects, ())
        self.assertEqual(container._parent_containers, ())

        self.assertEqual(container._container_child_objects, ())
        self.assertEqual(container._data_child_objects, ())

        self.assertEqual(container._repr_pretty_containers, ())

        self.assertEqual(container._child_objects, ())

        self.assertEqual(container._container_child_containers, ())
        self.assertEqual(container._data_child_containers, ())
        self.assertEqual(container._child_containers, ())

        self.assertEqual(container._child_objects, ())
        self.assertEqual(container._child_containers, ())

        self.assertEqual(container._single_children, ())
        self.assertEqual(container.data_children, ())
        self.assertEqual(container.container_children, ())
        self.assertEqual(container.children, ())
        self.assertEqual(container.parents, ())

        self.assertEqual(container.data_children_recur, ())
        self.assertEqual(container.container_children_recur, ())
        self.assertEqual(container.children_recur, ())

        self.assertEqual(container.filter(test=1), [])
        self.assertEqual(container.filter(data=True, container=False, test=1), [])
        self.assertEqual(container.filter(data=False, container=False, test=1), [])
        self.assertEqual(container.filter(data=True, container=True, test=1), [])
        self.assertEqual(container.filter(data=False, container=True, test=1), [])

        self.assertEqual(container.size, {})

        container.check_relationships()

    def test_filter(self):
        container = Container()
        self.assertRaises(TypeError, container.filter, "foo")

    def test_filter_results(self):
        """
        Tests FilterConditions correct results
        """
        self.assertEqual(self.st1.annotations, self.seg.filter(test=filters.Equals(5))[0].annotations)
        self.assertEqual(self.st1.annotations, self.seg.filter(test=filters.LessThan(6))[0].annotations)
        self.assertEqual(self.st1.annotations, self.seg.filter(test=filters.GreaterThan(4))[0].annotations)
        self.assertEqual(self.st1.annotations, self.seg.filter(test=filters.IsNot(1))[0].annotations)
        self.assertEqual(self.st1.annotations, self.seg.filter(test=filters.IsIn([1, 2, 5]))[0].annotations)
        self.assertEqual(self.st1.annotations, self.seg.filter(test=filters.InRange(1, 5))[0].annotations)
        self.assertEqual(self.st1.annotations, self.seg.filter(test=filters.GreaterThanOrEquals(5))[0].annotations)
        self.assertEqual(self.st1.annotations, self.seg.filter(test=filters.LessThanOrEquals(5))[0].annotations)

    def test_filter_equal(self):
        """
        Tests FilterCondition object "Equals".
        """
        self.assertEqual(1, len(self.seg.filter(test=filters.Equals(5))))
        self.assertEqual(0, len(self.seg.filter(test=filters.Equals(1))))
        self.assertEqual(1, len(self.seg.filter({"list": filters.Equals([1, 2])})))
        self.assertEqual(1, len(self.seg.filter(dict=filters.Equals({"key": 5}))))

    def test_filter_is_not(self):
        """
        Tests FilterCondition object "IsNot".
        """
        self.assertEqual(2, len(self.seg.filter(test=filters.IsNot(1))))
        self.assertEqual(1, len(self.seg.filter(test=filters.IsNot(5))))
        self.assertEqual(0, len(self.seg.filter([{"test": filters.IsNot(5)}, {"test": filters.IsNot(6)}])))

    def test_filter_less_than(self):
        """
        Tests FilterCondition object "LessThan".
        """
        self.assertEqual(0, len(self.seg.filter(test=filters.LessThan(5))))
        self.assertEqual(1, len(self.seg.filter(test=filters.LessThan(6))))
        self.assertEqual(2, len(self.seg.filter(test=filters.LessThan(7))))

    def test_filter_less_than_equal(self):
        """
        Tests FilterCondition object "LessThanEquals".
        """
        self.assertEqual(0, len(self.seg.filter(test=filters.LessThanOrEquals(4))))
        self.assertEqual(1, len(self.seg.filter(test=filters.LessThanOrEquals(5))))
        self.assertEqual(2, len(self.seg.filter(test=filters.LessThanOrEquals(6))))

    def test_filter_greater_than(self):
        """
        Tests FilterCondition object "GreaterThan".
        """
        self.assertEqual(0, len(self.seg.filter(test=filters.GreaterThan(6))))
        self.assertEqual(1, len(self.seg.filter(test=filters.GreaterThan(5))))
        self.assertEqual(2, len(self.seg.filter(test=filters.GreaterThan(4))))

    def test_filter_greater_than_equal(self):
        """
        Tests FilterCondition object "GreaterThanEquals".
        """
        self.assertEqual(0, len(self.seg.filter(test=filters.GreaterThanOrEquals(7))))
        self.assertEqual(1, len(self.seg.filter(test=filters.GreaterThanOrEquals(6))))
        self.assertEqual(2, len(self.seg.filter(test=filters.GreaterThanOrEquals(5))))

    def test_filter_is_in(self):
        """
        Tests FilterCondition object "IsIn".
        """
        # list
        self.assertEqual(0, len(self.seg.filter(test=filters.IsIn([4, 7, 10]))))
        self.assertEqual(1, len(self.seg.filter(test=filters.IsIn([5, 7, 10]))))
        self.assertEqual(2, len(self.seg.filter(test=filters.IsIn([5, 6, 10]))))
        # tuple
        self.assertEqual(0, len(self.seg.filter(test=filters.IsIn((4, 7, 10)))))
        self.assertEqual(1, len(self.seg.filter(test=filters.IsIn((5, 7, 10)))))
        self.assertEqual(2, len(self.seg.filter(test=filters.IsIn((5, 6, 10)))))
        # set
        self.assertEqual(0, len(self.seg.filter(test=filters.IsIn({4, 7, 10}))))
        self.assertEqual(1, len(self.seg.filter(test=filters.IsIn({5, 7, 10}))))
        self.assertEqual(2, len(self.seg.filter(test=filters.IsIn({5, 6, 10}))))

    def test_filter_in_range(self):
        """
        Tests FilterCondition object "InRange".
        """
        with self.assertRaises(ValueError):
            self.seg.filter(test=filters.InRange("wrong", 6, False, False))
        self.assertEqual(2, len(self.seg.filter(test=filters.InRange(5, 6, False, False))))
        self.assertEqual(1, len(self.seg.filter(test=filters.InRange(5, 6, True, False))))
        self.assertEqual(1, len(self.seg.filter(test=filters.InRange(5, 6, False, True))))
        self.assertEqual(0, len(self.seg.filter(test=filters.InRange(5, 6, True, True))))

    def test_filter_filter_consistency(self):
        """
        Tests old functionality with new filter method.
        """
        self.assertEqual(2, len(self.seg.filter({"test": filters.Equals(5), "filt": filters.Equals(6)})))
        self.assertEqual(0, len(self.seg.filter([{"test": filters.Equals(5)}, {"filt": filters.Equals(6)}])))
        self.assertEqual(1, len(self.seg.filter(name="st_num_1")))


class Test_Container_merge(unittest.TestCase):
    """
    TestCase to make sure merge method works
    """

    def setUp(self):
        self.name1 = "a container 1"
        self.name2 = "a container 2"
        self.description1 = "this is a test 1"
        self.description2 = "this is a test 2"
        self.cont1 = Container(name=self.name1, description=self.description1)
        self.cont2 = Container(name=self.name2, description=self.description2)

    def test_merge__dict(self):
        self.cont1.annotations = {"val1": 1, "val2": 2.2, "val3": "test1"}
        self.cont2.annotations = {"val2": 2.2, "val3": "test2", "val4": [4, 4.4], "val5": True}

        ann1 = self.cont1.annotations
        ann1c = self.cont1.annotations.copy()
        ann2c = self.cont2.annotations.copy()

        targ = {"val1": 1, "val2": 2.2, "val3": "test1;test2", "val4": [4, 4.4], "val5": True}

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
        self.cont1.annotations = {"val1": 1, "val2": 2.2, "val3": "tester"}
        self.cont2.annotations = {"val3": False, "val4": [4, 4.4], "val5": True}
        self.cont1.merge(self.cont2)
        self.assertEqual(
            self.cont1.annotations, {"val1": 1, "val2": 2.2, "val3": "MERGE CONFLICT", "val4": [4, 4.4], "val5": True}
        )

    def test_merge__unmergable_unequal_AssertionError(self):
        self.cont1.annotations = {"val1": 1, "val2": 2.2, "val3": True}
        self.cont2.annotations = {"val3": False, "val4": [4, 4.4], "val5": True}
        self.cont1.merge(self.cont2)
        self.assertEqual(
            self.cont1.annotations, {"val1": 1, "val2": 2.2, "val3": "MERGE CONFLICT", "val4": [4, 4.4], "val5": True}
        )


@unittest.skipUnless(HAVE_IPYTHON, "requires IPython")
class Test_pprint(unittest.TestCase):
    def test__pretty(self):
        name = "an object"
        description = "this is a test"
        obj = Container(name=name, description=description)
        res = pretty(obj)
        targ = f"Container with  name: '{name}' description: '{description}'"
        self.assertEqual(res, targ)


if __name__ == "__main__":
    unittest.main()
