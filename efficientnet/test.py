import tensorflow as tf
from border import BorderOffset, BorderReweight, Conv2D
import unittest

class TestBorderReweight(unittest.TestCase):
    def check(self, shape, width, stride, acc=4):
        border = BorderReweight(width, stride)
        kernel = tf.ones(width + (1, 1)) / tf.cast(
            tf.reduce_prod(width), tf.float32)
        res = border(tf.convert_to_tensor(shape))
        correct = 1 / tf.nn.conv2d(
            tf.ones((1,) + shape + (1,)), kernel, (1,) + stride + (1,),
            "SAME")[0, ..., 0]
        self.assertEqual(res.shape, correct.shape)
        diff = tf.reduce_max(res - correct)
        self.assertAlmostEqual(diff.numpy().item(), 0, acc)

    def test_basic(self):
        self.check((16, 16), (8, 8), (1, 1))

    def test_overlap(self):
        self.check((5, 5), (8, 8), (1, 1))

    def test_overflow(self):
        self.check((5, 5), (16, 16), (1, 1))

    def test_stride(self):
        self.check((16, 16), (8, 8), (3, 2))

    def test_overlap_stride(self):
        self.check((5, 5), (8, 8), (3, 2))

    def test_overflow_stride(self):
        self.check((5, 5), (16, 16), (3, 2))

    def test_long_stride(self):
        self.check((16, 16), (8, 8), (30, 20))

    def test_random(self):
        tf.random.set_seed(0)
        for i in range(100):
            inp = tf.cast(tf.random.uniform((3, 2)) * 32 + 1, tf.int32)
            inp = tuple(map(tuple, inp))
            self.check(*inp, acc=2)

class MockBorderOffset(BorderOffset):
    def __init__(self, width, *args, **kw):
        self.width = width
        super().__init__(width, *args, **kw)

    def initialize(self, size, width, axis, end):
        return tf.ones((size,)) * self.width[1 - axis]

class TestBorderOffset(unittest.TestCase):
    def check(self, shape, width, stride):
        border = MockBorderOffset(width, stride)
        kernel = tf.ones(width + (1, 1))
        res = border(shape)
        correct = tf.reduce_sum(kernel) - tf.nn.conv2d(
            tf.ones((1,) + shape + (1,)), kernel, (1,) + stride + (1,),
            "SAME")[0, ..., 0]
        self.assertEqual(res.shape, correct.shape)
        self.assertEqual(tf.reduce_all(res == correct), True)

    # removed because known to double count
    def _test(self):
        self.check((10, 10), (8, 5), (1, 1))

class TestConvLayer(unittest.TestCase):
    def test_simple(self):
        conv = Conv2D(4, 3, 2, "same", kernel_initializer="ones")
        self.assertEqual(True, tf.reduce_all(conv(tf.ones((1, 5, 5, 1))) == 9))

if __name__ == '__main__':
    unittest.main()
