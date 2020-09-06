import tensorflow as tf
from border import BorderOffset, BorderReweight

class Conv:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.padding == "same" and all(
            i == 1 for i in self.dilation_rate)
        self.border_reweight = BorderReweight(
            self.kernel_size, self.strides, self.rank)
        self.border_offset = BorderOffset(
            self.kernel_size, self.strides, self.rank, self.use_bias)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        super().build(input_shape)
        input_shape = input_shape[2:] if self._channels_first else \
            input_shape[1:-1]
        self.built_reweight = tf.expand_dims(self.border_reweight(
            input_shape)[tf.newaxis, ...], self._get_channel_axis())
        self.built_offset = tf.expand_dims(self.border_offset(
            input_shape)[tf.newaxis, ...], self._get_channel_axis())

    def call(self, inputs):
        return self.built_reweight * super().call(inputs) + self.built_offset

class Conv1D(Conv, tf.keras.layers.Conv1D):
    pass

class Conv2D(Conv, tf.keras.layers.Conv2D):
    pass

class Conv3D(Conv, tf.keras.layers.Conv3D):
    pass

class DepthwiseConv2D(Conv, tf.keras.layers.DepthwiseConv2D):
    pass
