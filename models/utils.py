import tensorflow as tf

def tpu_prep(f):
    return lambda im: tf.cast(tf.transpose(f(im), [1, 2, 3, 0]), tf.bfloat16)

class TPUBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, fused=False, name="BatchNormalization", **kwargs):
        assert tf.distribute.in_cross_replica_context()
        if fused in (True, None):
            raise ValueError('fused batch norm not supported across groups')
        super().__init__(fused=fused, name=name, **kwargs)

    def _group_average(self, tensor, shards, group_size):
        assignments = [[j for j in range(shards) if j // group_size == i]
                       for i in range(group_size)]
        return tf.raw_ops.CrossReplicaSum(tensor, assignments) * tf.cast(
            group_size / shards, tensor.dtype)

    def _moments(self, inputs, axes, keep_dims):
        means, variances = super()._moments(inputs, axes, keep_dims=keep_dims)
        shards = tf.distribute.get_strategy().num_replicas_in_sync or 1

        if shards > 8:
            # Var[X] = E[X ^ 2] - E[X] ^ 2.
            group_size = max(8, shards // 8)
            group_mean = self._group_mean(means, shards, group_size)
            l2sq = self._group_mean(variances + means ** 2, shards, group_size)
            return (group_mean, l2sq - group_mean ** 2)
        else:
            return (means, variances)

class Conv2D(tf.keras.layers.Conv2D):
    def call(self, inputs):
        single_inference = self.data_format == "channels_first" and \
            inputs.shape[0] == 1 and self.kernel_size == (1, 1)
        if not single_inference:
            return super().call(inputs)
        shape = tf.shape(inputs)
        flat = tf.reshape(inputs, [shape[1], -1])
        scaled = tf.transpose(tf.squeeze(self.kernel, (2, 3))) @ flat
        target = tf.concat(((1, self.filters), shape[2:]), 0)
        res = tf.reshape(scaled, target)

        if self.use_bias:
            res = tf.nn.bias_add(res, self.bias, data_format='NCHW')
        if self.activation is not None:
            res = self.activation(res)
        return res
