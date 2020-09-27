import tensorflow as tf
import argparse, os
from functools import partial, wraps
from datetime import datetime

class HelpFormatter(argparse.HelpFormatter):
    def _format_args(self, action, default_metavar):
        if hasattr(action, "format_meta"):
            return action.format_meta(self._metavar_formatter(
                action, default_metavar))
        else:
            return super()._format_args(action, default_metavar)

class Duplicate:
    def __init__(self, *cls):
        assert len(cls) > 0
        self.__dict__["cls"] = cls

    def __getattr__(self, key):
        res = getattr(self.cls[0], key)
        if callable(res):
            @wraps(res)
            def wrapper(*args, **kwargs):
                out = res(*args, **kwargs)
                for cls in self.cls[1:]:
                    getattr(cls, key)(*args, **kwargs) 
                return out
            return wrapper
        return res

    def __setattr__(self, key, value):
        for cls in self.cls:
            setattr(cls, key, value)

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, formatter_class=HelpFormatter, **kwargs):
        kwargs = {**kwargs, "formatter_class": formatter_class}
        super().__init__(*args, **kwargs)
        self._group, self.copy = None, lambda: self.__class__(*args, **kwargs)

    @property
    def group(self):
        if self._group is None:
            self._group = self.copy()
            self._subgroup = Duplicate(self, self._group)
        return self._subgroup

    def parse_known_args(self, args=None, namespace=None):
        res, args = super().parse_known_args(args, namespace)
        if self._group is not None:
            res, args = self._group.parse_known_args(args, res)
        self._group = None

        if hasattr(res, "call"):
            res.caller = partial(res.call, **{i: j for i, j in vars(
                res).items() if i not in ("call", "caller")})
        return res, args

relpath = lambda *args: os.path.join(
    os.path.dirname(os.path.abspath(__file__)), *args)

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
        def __init__(self, option_strings, dest, **kwargs):
            super().__init__(option_strings, dest, nargs="*", **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            if not minimum <= len(values) <= maximum:
                raise argparse.ArgumentTypeError(
                    f'argument "{self.dest}" requires between {minimum} and '
                    f'{maximum} arguments')
            setattr(namespace, self.dest, values)

        def format_meta(self, metavars):
            metavars = metavars(maximum)
            formatted = ' %s' * minimum + ' [%s' * (maximum - minimum) \
                + ']' * maximum
            return formatted[1:] % metavars

    return RequiredLength

def PresetFlag(*preset):
    class PresetFlag(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, list(preset) + values)
    return PresetFlag

def ExtendCLI(f):
    class ExtendCLI(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            setattr(namespace, self.dest, f(parser.group, value))
    return ExtendCLI

strftime = lambda: datetime.today().strftime("%Y_%m_%d_%H_%M_%S")

def cli_call(*f):
    parser = ArgumentParser()
    for i in f:
        i(parser)

    return parser.parse_args().caller()

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
        shape = tf.shape(inputs)
        single_inference = self.data_format == "channels_first" and \
            shape[0] == 1 and self.kernel_size == (1, 1)
        if not single_inference:
            return super().call(inputs)
        flat = tf.reshape(inputs, [shape[1], -1])
        scaled = tf.transpose(tf.squeeze(self.kernel, (2, 3))) @ flat
        target = tf.concat(((1, self.filters), shape[2:]), 0)
        res = tf.reshape(scaled, target)

        if self.use_bias:
            res = tf.nn.bias_add(res, self.bias, data_format='NCHW')
        if self.activation is not None:
            res = self.activation(res)
        return res
