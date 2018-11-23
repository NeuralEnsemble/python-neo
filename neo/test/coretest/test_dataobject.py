import copy

import numpy as np
import unittest

from neo.core.dataobject import DataObject, _normalize_array_annotations, ArrayDict


class Test_DataObject(unittest.TestCase):
    def test(self):
        pass


class Test_array_annotations(unittest.TestCase):
    def test_check_arr_ann(self):
        # DataObject instance that handles checking
        datobj = DataObject([1, 2])  # Inherits from Quantity, so some data is required

        # Correct annotations
        arr1 = np.asarray(["ABC", "DEF"])
        arr2 = np.asarray([3, 6])
        corr_ann = {'anno1': arr1, 'anno2': arr2}

        corr_ann_copy = copy.deepcopy(corr_ann)

        # Checking correct annotations should work fine
        corr_ann = _normalize_array_annotations(corr_ann, datobj._get_arr_ann_length())

        # Make sure the annotations have not been altered
        self.assertSequenceEqual(corr_ann.keys(), corr_ann_copy.keys())
        self.assertTrue((corr_ann['anno1'] == corr_ann_copy['anno1']).all())
        self.assertTrue((corr_ann['anno2'] == corr_ann_copy['anno2']).all())

        # Now creating incorrect inputs:

        # Nested dict
        nested_ann = {'anno1': {'val1': arr1}, 'anno2': {'val2': arr2}}
        with self.assertRaises(ValueError):
            nested_ann = _normalize_array_annotations(nested_ann, datobj._get_arr_ann_length())

        # Containing None
        none_ann = corr_ann_copy
        # noinspection PyTypeChecker
        none_ann['anno2'] = None
        with self.assertRaises(ValueError):
            none_ann = _normalize_array_annotations(none_ann, datobj._get_arr_ann_length())

        # Multi-dimensional arrays in annotations
        multi_dim_ann = copy.deepcopy(corr_ann)
        multi_dim_ann['anno2'] = multi_dim_ann['anno2'].reshape(1, 2)
        with self.assertRaises(ValueError):
            multi_dim_ann = _normalize_array_annotations(multi_dim_ann,
                                                         datobj._get_arr_ann_length())

        # Wrong length of annotations
        len_ann = corr_ann
        len_ann['anno1'] = np.asarray(['ABC', 'DEF', 'GHI'])
        with self.assertRaises(ValueError):
            len_ann = _normalize_array_annotations(len_ann, datobj._get_arr_ann_length())

        # Scalar as array annotation raises Error if len(datobj)!=1
        scalar_ann = copy.deepcopy(corr_ann)
        # noinspection PyTypeChecker
        scalar_ann['anno2'] = 3
        with self.assertRaises(ValueError):
            scalar_ann = _normalize_array_annotations(scalar_ann, datobj._get_arr_ann_length())

        # But not if len(datobj) == 1, then it's wrapped into an array
        # noinspection PyTypeChecker
        scalar_ann['anno1'] = 'ABC'
        datobj2 = DataObject([1])
        scalar_ann = _normalize_array_annotations(scalar_ann, datobj2._get_arr_ann_length())
        self.assertIsInstance(scalar_ann['anno1'], np.ndarray)
        self.assertIsInstance(scalar_ann['anno2'], np.ndarray)

        # Lists are also made to np.ndarrays
        list_ann = {'anno1': [3, 6], 'anno2': ['ABC', 'DEF']}
        list_ann = _normalize_array_annotations(list_ann, datobj._get_arr_ann_length())
        self.assertIsInstance(list_ann['anno1'], np.ndarray)
        self.assertIsInstance(list_ann['anno2'], np.ndarray)

    def test_implicit_dict_check(self):
        # DataObject instance that handles checking
        datobj = DataObject([1, 2])  # Inherits from Quantity, so some data is required

        # Correct annotations
        arr1 = np.asarray(["ABC", "DEF"])
        arr2 = np.asarray([3, 6])
        corr_ann = {'anno1': arr1, 'anno2': arr2}

        corr_ann_copy = copy.deepcopy(corr_ann)

        # Implicit checks when setting item in dict directly
        # Checking correct annotations should work fine
        datobj.array_annotations['anno1'] = arr1
        datobj.array_annotations.update({'anno2': arr2})

        # Make sure the annotations have not been altered
        self.assertTrue((datobj.array_annotations['anno1'] == corr_ann_copy['anno1']).all())
        self.assertTrue((datobj.array_annotations['anno2'] == corr_ann_copy['anno2']).all())

        # Now creating incorrect inputs:

        # Nested dict
        nested_ann = {'anno1': {'val1': arr1}, 'anno2': {'val2': arr2}}
        with self.assertRaises(ValueError):
            datobj.array_annotations['anno1'] = {'val1': arr1}

        # Containing None
        none_ann = corr_ann_copy
        # noinspection PyTypeChecker
        none_ann['anno2'] = None
        with self.assertRaises(ValueError):
            datobj.array_annotations['anno1'] = None

        # Multi-dimensional arrays in annotations
        multi_dim_ann = copy.deepcopy(corr_ann)
        multi_dim_ann['anno2'] = multi_dim_ann['anno2'].reshape(1, 2)
        with self.assertRaises(ValueError):
            datobj.array_annotations.update(multi_dim_ann)

        # Wrong length of annotations
        len_ann = corr_ann
        len_ann['anno1'] = np.asarray(['ABC', 'DEF', 'GHI'])
        with self.assertRaises(ValueError):
            datobj.array_annotations.update(len_ann)

        # Scalar as array annotation raises Error if len(datobj)!=1
        scalar_ann = copy.deepcopy(corr_ann)
        # noinspection PyTypeChecker
        scalar_ann['anno2'] = 3
        with self.assertRaises(ValueError):
            datobj.array_annotations.update(scalar_ann)

        # But not if len(datobj) == 1, then it's wrapped into an array
        # noinspection PyTypeChecker
        scalar_ann['anno1'] = 'ABC'
        datobj2 = DataObject([1])
        datobj2.array_annotations.update(scalar_ann)
        self.assertIsInstance(datobj2.array_annotations['anno1'], np.ndarray)
        self.assertIsInstance(datobj2.array_annotations['anno2'], np.ndarray)

        # Lists are also made to np.ndarrays
        list_ann = {'anno1': [3, 6], 'anno2': ['ABC', 'DEF']}
        datobj.array_annotations.update(list_ann)
        self.assertIsInstance(datobj.array_annotations['anno1'], np.ndarray)
        self.assertIsInstance(datobj.array_annotations['anno2'], np.ndarray)

    def test_array_annotate(self):
        # Calls _check_array_annotations, so no need to test for these Errors here
        datobj = DataObject([2, 3, 4])
        arr_ann = {'anno1': [3, 4, 5], 'anno2': ['ABC', 'DEF', 'GHI']}

        # Pass annotations
        datobj.array_annotate(**arr_ann)

        # Make sure they are correct
        self.assertTrue((datobj.array_annotations['anno1'] == np.array([3, 4, 5])).all())
        self.assertTrue(
            (datobj.array_annotations['anno2'] == np.array(['ABC', 'DEF', 'GHI'])).all())
        self.assertIsInstance(datobj.array_annotations, ArrayDict)

    def test_arr_anns_at_index(self):
        # Get them, test for desired type and size, content
        datobj = DataObject([1, 2, 3, 4])
        arr_ann = {'anno1': [3, 4, 5, 6], 'anno2': ['ABC', 'DEF', 'GHI', 'JKL']}
        datobj.array_annotate(**arr_ann)

        # Integer as index
        ann_int = datobj.array_annotations_at_index(1)
        self.assertEqual(ann_int, {'anno1': 4, 'anno2': 'DEF'})
        # Negative integer as index
        ann_int_back = datobj.array_annotations_at_index(-2)
        self.assertEqual(ann_int_back, {'anno1': 5, 'anno2': 'GHI'})

        # Slice as index
        ann_slice = datobj.array_annotations_at_index(slice(1, 3))
        self.assert_((ann_slice['anno1'] == np.array([4, 5])).all())
        self.assert_((ann_slice['anno2'] == np.array(['DEF', 'GHI'])).all())

        # Slice from beginning to end
        ann_slice_all = datobj.array_annotations_at_index(slice(0, None))
        self.assert_((ann_slice_all['anno1'] == np.array([3, 4, 5, 6])).all())
        self.assert_((ann_slice_all['anno2'] == np.array(['ABC', 'DEF', 'GHI', 'JKL'])).all())

        # Make sure that original object is edited when editing extracted array_annotations
        ann_slice_all['anno1'][2] = 10
        self.assertEqual(datobj.array_annotations_at_index(2)['anno1'], 10)
