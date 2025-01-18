import tensorflow as tf
import argparse, os, gc
from functools import partial, wraps
from datetime import datetime
from inspect import signature

class HelpFormatter(argparse.HelpFormatter):
    def _format_args(self, action, default_metavar):
        if hasattr(action, "format_meta"):
            return action.format_meta(self._metavar_formatter(
                action, default_metavar))
        else:
            return super()._format_args(action, default_metavar)

# TODO: since this was written, Python started supporting Ellipsis as defaults
class Default:
    pass

def cli_builder(f):
    params = signature(f).parameters
    @wraps(f)
    def wrapper(*args, **kw):
        args = tuple(param.default if arg is Default else arg
                     for arg, param in zip(args, params.values()))
        kw = {k: params[k].default if v is Default and k in params else v
              for k, v in kw.items()}
        return f(*args, **kw)
    return wrapper

class CallParser:
    def __init__(self, *args, formatter_class=HelpFormatter,
                 argument_default=Default, **kw):
        super().__init__(*args, formatter_class=formatter_class,
                         argument_default=argument_default, **kw)

    def parse_known_args(self, args=None, namespace=None):
        res, args = super().parse_known_args(args, namespace)

        if hasattr(res, "call"):
            res.caller = partial(res.call, **{i: j for i, j in vars(
                res).items() if i not in ("call", "caller")})
        return res, args

class ArgumentParser(CallParser, argparse.ArgumentParser):
    pass

relpath = lambda *args: os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, *args))

class NoStrategy:
    def __init__(self):
        self.num_replicas_in_sync = 1

    def scope(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def RequiredLength(minimum, maximum):
    class RequiredLength(argparse.Action):
        def __init__(self, option_strings, dest, **kw):
            super().__init__(option_strings, dest, nargs="*", **kw)

        def __call__(self, parser, namespace, values, option_string=None):
            if not minimum <= len(values) <= maximum:
                raise argparse.ArgumentTypeError(
                    f'argument "{self.dest}" requires between {minimum} and '
                    f'{maximum} arguments')
            setattr(namespace, self.dest, values)

        def format_meta(self, metavars):
            metavars = metavars(maximum)
            formatted = ' %s' * minimum + ' [%s' * (maximum - minimum) \
                + ']' * minimum
            return formatted[1:] % metavars

    return RequiredLength

def PresetFlag(*preset):
    class PresetFlag(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if self.default is not None or len(values) == 0:
                setattr(namespace, self.dest, list(preset) + self.default)
                return
            setattr(namespace, self.dest, list(preset) + values)
    return PresetFlag

strftime = lambda: datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
helper = lambda parser: lambda: parser.parse_args(["-h"])

def parse_strategy(distribute):
    distribute, *devices = distribute or [None]
    assert not distribute or hasattr(tf.distribute, distribute)
    devices = None if len(devices) == 0 else \
        devices[0] if distribute == "OneDeviceStrategy" else devices

    tpu = distribute == "TPUStrategy"
    if tpu:
        devices = tf.distribute.cluster_resolver.TPUClusterResolver(*devices)
        tf.config.experimental_connect_to_cluster(devices)
        tf.tpu.experimental.initialize_tpu_system(devices)

    distribute = NoStrategy() if distribute is None else distribute
    distribute = getattr(tf.distribute, distribute)(devices) \
        if type(distribute) is str else distribute

    return distribute, tpu

def tpu_prep(f):
    def body(im, mask=None):
        im, mask = f(im, mask)
        return tf.cast(im, tf.bfloat16), mask
    return body

class TPUBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, fused=False, name="BatchNormalization", **kw):
        assert tf.distribute.in_cross_replica_context()
        if fused in (True, None):
            raise ValueError('fused batch norm not supported across groups')
        super().__init__(fused=fused, name=name, **kw)

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

# specialized for fast inference with MBConv
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

class WarmedExponential(tf.keras.callbacks.Callback):
    def __init__(self, scale, units, warmup, decay, step=None, freq=64):
        self.scale, self.units, self.warmup, self.decay, self.freq = scale, \
            units, warmup, decay, freq
        self.step = tf.Variable(0, dtype=tf.int32, name="step") \
            if step is None else step

    def on_train_batch_begin(self, batch, logs=None):
        self.step.assign_add(1)
        if self.step % self.freq == 1:
            x = (tf.cast(self.step, tf.float32) + self.freq / 2) / self.units
            lr = self.scale * x / self.warmup if x < self.warmup else \
                self.scale * self.decay ** (x - self.warmup)
            self.model.optimizer.learning_rate.assign(lr)

class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'learning_rate': self.model.optimizer.learning_rate})
        super().on_epoch_end(epoch, logs)

class Checkpointer(tf.keras.callbacks.Callback):
    def __init__(self, prefix, epoch, root=None, **kw):
        self.prefix, self.epoch = prefix, epoch
        self.ckpt = tf.train.Checkpoint(root, epoch=epoch, **kw)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch.assign_add(1)
        self.ckpt.save(self.prefix)

    def restore(self, base):
        path = tf.train.latest_checkpoint(base)
        self.ckpt.restore(path)
        return path

def perimeter_pixel(radius, i):
    """
    https://g.co/gemini/share/f65c2c57321b

    Given a random number (i) from 0 to 8 times the radius of a square
    centered at pixel (0, 0), find the coordinates of a uniformly
    distributed pixel along the perimeter using tf.where.

    Args:
        radius: The radius of the square.
        i: The random number representing a position along the perimeter.

    Returns:
        A tuple representing the (x, y) coordinates of the pixel.
    """
    side_length = 2 * radius

    x = tf.where(i < side_length, i - radius,
            tf.where(i < 2 * side_length, radius, tf.where(
                i < 3 * side_length, radius - (i - 2 * side_length), -radius)))

    y = tf.where(i < side_length, radius,
            tf.where(i < 2 * side_length, radius - (i - side_length), tf.where(
                i < 3 * side_length, -radius, (i - 3 * side_length) - radius)))

    return x, y

def serialize_tensor_features(values):
    res = {}
    for k, value in values.items():
        serialized_nonscalar = tf.io.serialize_tensor(value)
        res[k] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[serialized_nonscalar.numpy()]))
    return tf.train.Example(features=tf.train.Features(feature=res))

