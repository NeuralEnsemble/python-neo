"""
Tests of the neo.core.objectlist.ObjectList class
"""

import unittest

import quantities as pq

from neo.core import AnalogSignal, Event, Group, Segment
from neo.core.objectlist import ObjectList


def make_signal(value=1.0):
    return AnalogSignal([[value], [value + 1], [value + 2]] * pq.mV, sampling_rate=1 * pq.kHz)


class TestObjectList(unittest.TestCase):
    def setUp(self):
        self.segment = Segment()
        self.objlist = ObjectList(AnalogSignal, parent=self.segment)
        self.a, self.b, self.c = make_signal(1.0), make_signal(10.0), make_signal(100.0)

    def test_append_sets_parent(self):
        self.objlist.append(self.a)
        self.assertEqual(len(self.objlist), 1)
        self.assertIs(self.a.segment, self.segment)

    def test_append_duplicate(self):
        self.objlist.append(self.a)
        self.assertRaises(ValueError, self.objlist.append, self.a)
        self.assertEqual(len(self.objlist), 1)

    def test_append_equal_but_distinct_object(self):
        # membership is by identity, so an object with the same data is a different object
        self.objlist.append(self.a)
        self.objlist.append(make_signal(1.0))
        self.assertEqual(len(self.objlist), 2)

    def test_append_wrong_type(self):
        self.assertRaises(TypeError, self.objlist.append, Event())

    def test_extend_from_generator(self):
        self.objlist.extend(sig for sig in (self.a, self.b))
        self.assertEqual(len(self.objlist), 2)
        self.assertIs(self.objlist[1], self.b)

    def test_iadd_from_generator(self):
        self.objlist += (sig for sig in (self.a, self.b))
        self.assertEqual(len(self.objlist), 2)

    def test_extend_duplicate_within_iterable(self):
        self.assertRaises(ValueError, self.objlist.extend, [self.a, self.a])
        self.assertEqual(len(self.objlist), 0)
        self.assertIsNone(self.a.segment)

    def test_extend_duplicate_of_existing_item(self):
        self.objlist.append(self.a)
        self.assertRaises(ValueError, self.objlist.extend, [self.b, self.a])
        self.assertEqual(len(self.objlist), 1)
        self.assertIsNone(self.b.segment)

    def test_extend_is_atomic(self):
        self.assertRaises(TypeError, self.objlist.extend, [self.a, Event()])
        self.assertEqual(len(self.objlist), 0)
        self.assertIsNone(self.a.segment)

    def test_contains(self):
        self.objlist.extend([self.a, self.b])
        self.assertIn(self.a, self.objlist)
        self.assertIn(self.b, self.objlist)
        self.assertNotIn(self.c, self.objlist)
        self.assertNotIn(make_signal(10.0), self.objlist)

    def test_index(self):
        self.objlist.extend([self.a, self.b, self.c])
        self.assertEqual(self.objlist.index(self.c), 2)
        self.assertEqual(self.objlist.index(self.c, 1), 2)
        self.assertEqual(self.objlist.index(self.c, -1), 2)
        self.assertRaises(ValueError, self.objlist.index, self.a, 1)
        self.assertRaises(ValueError, self.objlist.index, make_signal(10.0))

    def test_count(self):
        self.objlist.extend([self.a, self.b])
        self.assertEqual(self.objlist.count(self.b), 1)
        self.assertEqual(self.objlist.count(make_signal(10.0)), 0)

    def test_remove(self):
        self.objlist.extend([self.a, self.b, self.c])
        self.objlist.remove(self.b)
        self.assertEqual(list(self.objlist), [self.a, self.c])
        self.assertRaises(ValueError, self.objlist.remove, self.b)

    def test_append_after_removal(self):
        self.objlist.extend([self.a, self.b])
        self.objlist.remove(self.a)
        self.objlist.append(self.a)
        self.assertIs(self.objlist.pop(), self.a)
        self.objlist.append(self.a)
        self.objlist.clear()
        self.objlist.append(self.a)
        self.assertEqual(len(self.objlist), 1)

    def test_setitem(self):
        self.objlist.extend([self.a, self.b])
        self.objlist[1] = self.c
        self.assertIs(self.objlist[1], self.c)
        self.assertIs(self.c.segment, self.segment)

    def test_setitem_same_object(self):
        self.objlist.extend([self.a, self.b])
        self.objlist[1] = self.b
        self.assertEqual(list(self.objlist), [self.a, self.b])

    def test_setitem_duplicate(self):
        self.objlist.extend([self.a, self.b])
        with self.assertRaises(ValueError):
            self.objlist[1] = self.a
        self.assertEqual(list(self.objlist), [self.a, self.b])

    def test_setitem_wrong_type(self):
        self.objlist.append(self.a)
        with self.assertRaises(TypeError):
            self.objlist[0] = Event()

    def test_setitem_slice(self):
        self.objlist.extend([self.a, self.b])
        self.objlist[1:] = [self.b, self.c]
        self.assertEqual(list(self.objlist), [self.a, self.b, self.c])
        self.assertIs(self.c.segment, self.segment)

    def test_setitem_slice_duplicate(self):
        self.objlist.extend([self.a, self.b])
        with self.assertRaises(ValueError):
            self.objlist[1:] = [self.c, self.a]
        with self.assertRaises(ValueError):
            self.objlist[1:] = [self.c, self.c]
        self.assertEqual(list(self.objlist), [self.a, self.b])
        self.assertIsNone(self.c.segment)

    def test_setitem_slice_wrong_type(self):
        self.objlist.append(self.a)
        with self.assertRaises(TypeError):
            self.objlist[:] = [self.b, Event()]
        self.assertEqual(list(self.objlist), [self.a])

    def test_reverse(self):
        self.objlist.extend([self.a, self.b, self.c])
        self.objlist.reverse()
        self.assertEqual(list(self.objlist), [self.c, self.b, self.a])

    def test_without_parent(self):
        group = Group()
        objlist = ObjectList(AnalogSignal)
        objlist.extend([self.a, self.b])
        self.assertEqual(len(objlist), 2)
        self.assertIsNone(self.a.segment)
        group.analogsignals.append(self.c)
        self.assertIn(self.c, group.analogsignals)


if __name__ == "__main__":
    unittest.main()
