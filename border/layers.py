import tensorflow as tf
from .border import BorderOffset, BorderReweight

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.padding == "same" and all(
            i == 1 for i in self.dilation_rate)
        self.border_reweight = BorderReweight(
            self.kernel_size, self.strides, self.rank, self.register(True))
        if self.use_bias:
            self.border_offset = BorderOffset(
                self.kernel_size, self.strides, self.rank, self.register())

    def build(self, input_shape):
        super().build(input_shape)
        input_shape = input_shape[2:] if self._channels_first else \
            input_shape[1:-1]
        self.built_reweight = tf.expand_dims(self.border_reweight(
            input_shape)[tf.newaxis, ...], self._get_channel_axis())
        if self.use_bias:
            self.built_offset = tf.expand_dims(self.border_offset(
                input_shape)[tf.newaxis, ...], self._get_channel_axis())
        else:
            self.built_offset = 0.

    def call(self, inputs):
        return self.built_reweight * super().call(inputs) + self.built_offset

class Conv1D(BorderConv, tf.keras.layers.Conv1D):
    pass

class Conv2D(BorderConv, tf.keras.layers.Conv2D):
    pass

class Conv3D(BorderConv, tf.keras.layers.Conv3D):
    pass

class DepthwiseConv2D(BorderConv, tf.keras.layers.DepthwiseConv2D):
    pass
