import tensorflow as tf
from .border import BorderOffset, BorderReweight
from models.utils import Conv2D as SpecializedConv2D

class BorderConv:
    def register(self, kernel=False):
        args = (self.kernel_regularizer, self.kernel_constraint) \
            if kernel else (self.bias_regularizer, self.bias_constraint)
        added = dict(zip(("regularizer", "constraint"), args))
        def wrapper(initial, name):
            wrapped = lambda *args, **kwargs: initial
            return self.add_weight(
                name, tf.shape(initial), initial.dtype, wrapped, **added)
        return wrapper

    def __init__(self, *args, activation=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.padding == "same" and all(
            i == 1 for i in self.dilation_rate)
        self._activation = tf.keras.activations.get(activation)
        if self._activation is None:
            self._activation = lambda x: x

        self.small = bool(tf.reduce_all(self.kernel_size == tf.constant(1)))
        if self.small:
            return

        self.border_weight = BorderReweight(
            self.kernel_size, self.strides, self.rank, self.register(True))
        if self.use_bias:
            self.border_bias = BorderOffset(
                self.kernel_size, self.strides, self.rank, self.register())

    def _build(self, input_shape):
        input_shape = input_shape[2:] if self._channels_first else \
            input_shape[1:-1]
        weight = tf.expand_dims(self.border_weight(
            input_shape)[tf.newaxis, ...], self._get_channel_axis())
        bias = 0. if not self.use_bias else tf.expand_dims(self.border_bias(
            input_shape)[tf.newaxis, ...], self._get_channel_axis())
        return weight, bias

    def call(self, inputs):
        res = super().call(inputs)
        if self.small:
            return self._activation(res)

        weight, bias = self._build(tf.shape(inputs))
        return self._activation(weight * res + bias)

class Conv1D(BorderConv, tf.keras.layers.Conv1D):
    pass

class Conv2D(BorderConv, SpecializedConv2D):
    pass

class Conv3D(BorderConv, tf.keras.layers.Conv3D):
    pass

class DepthwiseConv2D(BorderConv, tf.keras.layers.DepthwiseConv2D):
    pass
