
import unittest

import numpy as np

import faiss
import example_external_module


class TestCustomIDSelector(unittest.TestCase):
    """ test if we can construct a custom IDSelector """

    def test_IDSelector(self):
        ids = example_external_module.IDSelectorModulo(3)
        self.assertFalse(ids.is_member(1))
        self.assertTrue(ids.is_member(3))


class TestArrayConversions(unittest.TestCase):

    def test_idx_array(self):
        tab = np.arange(10).astype('int64')
        new_sum = example_external_module.sum_of_idx(len(tab), faiss.swig_ptr(tab))
        self.assertEqual(new_sum, tab.sum())

    def do_array_test(self, ty):
        tab = np.arange(10).astype(ty)
        func = getattr(example_external_module, 'sum_of_' + ty)
        print("perceived type", faiss.swig_ptr(tab))
        new_sum = func(len(tab), faiss.swig_ptr(tab))
        self.assertEqual(new_sum, tab.sum())

    def test_sum_uint8(self):
        self.do_array_test('uint8')

    def test_sum_uint16(self):
        self.do_array_test('uint16')

    def test_sum_uint32(self):
        self.do_array_test('uint32')

    def test_sum_uint64(self):
        self.do_array_test('uint64')

    # this conversion does not work
    def test_sum_int8(self):
        self.do_array_test('int8')

    def test_sum_int16(self):
        self.do_array_test('int16')

    def test_sum_int32(self):
        self.do_array_test('int32')

    def test_sum_int64(self):
        self.do_array_test('int64')

    def test_sum_float32(self):
        self.do_array_test('float32')

    def test_sum_float64(self):
        self.do_array_test('float64')
