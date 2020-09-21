import tensorflow as tf
from collections import defaultdict
from functools import partial, wraps, update_wrapper
from inspect import signature, Parameter

class CondCall:
    def __init__(self, parent, f, bypass=False):
        assert type(bypass) == bool
        self.f, self.parent, self.bypass = f, parent, bypass
        update_wrapper(self, f)

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def cond(self, tensor, im, *args, **kwargs):
        if self.bypass:
            return self(im, *args, **kwargs)

        identity = self.parent.variables
        res, self.parent.variables = tf.cond(
            tensor, lambda: (self(im, *args, **kwargs), self.parent.variables),
            lambda: (im, identity))
        return res

class OpsList:
    def __init__(self, parent, ops, args, kwargs):
        self.parent = parent
        sub = [[j(*args, **kwargs) for j in i.ops] for i in ops]
        self.ops, self.objs, self.required = (sum(i, []) for i in zip(*sum((
            [(obj.ops.ops, obj.ops.objs, obj.ops.required) for obj in objs] + [
                ([op], [parent], [op.required])]
            for objs, op in zip(sub, ops)), [])))

        self.choosable = len(self) - sum(self.required)
        self.offset = tf.range(self.choosable) + tf.cumsum(tf.cast(
            self.required, tf.int32))[tf.math.logical_not(self.required)]

    def __getitem__(self, i):
        wrapped = partial(self.ops[i].caller, self.objs[i])
        wrapped = wraps(self.ops[i].call)(wrapped)
        wrapped = partial(wrapped, self.ops[i])
        return CondCall(self.parent, wrapped, self.required[i])

    def _sample(self, n, m): # chooses n out of the first m natural numbers
        return tf.random.uniform_candidate_sampler(
            tf.range(m, dtype=tf.int64)[tf.newaxis, :], m, n, True, m
        ).sampled_candidates

    def sample(self, n):
        assert 0 <= n < len(self)
        chosen = tf.gather(self.offset, self._sample(n, self.choosable))
        updates = tf.repeat(True, tf.shape(chosen))
        mask = tf.scatter_nd(chosen[:, tf.newaxis], updates, (len(self),))
        return (wraps(op)(partial(op.cond, mask[i]))
                for i, op in enumerate(self))

    def __len__(self):
        return len(self.ops)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    @property
    def tracking(self):
        return set(sum(([(obj, var) for var in op.track] for op, obj in zip(
            self.ops, self.objs)), []))

class Normalized:
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        def decorator(f):
            self.__init__(f)
            self.normal = self.sig.bind_partial(*args, **kwargs).arguments
            for k, v in self.normal.items():
                kind = self.sig.parameters[k]
                if kind == Parameter.VAR_POSITIONAL:
                    self.normal[k] = tuple(self.parse(v) for v in v)
                if kind == Parameter.VAR_KEYWORD:
                    self.normal[k] = {k: self.parse(v) for k, v in v}
                else:
                    self.normal[k] = self.parse(v)
            return wraps(f)(self)
        return decorator

    def __init__(self, f):
        self.f, self.sig = f, signature(f)

    def __call__(self, *args, **kwargs):
        bound = self.sig.bind(*args, **kwargs)
        bound.apply_defaults()
        bargs = bound.arguments
        for k, v in bargs.items():
            if k in self.normal and self.normal[k]:
                res = self.map(*self.normal[k], v)
                bargs[k] = res
        return self.f(*bound.args, **bound.kwargs)

    @classmethod
    def parse(cls, args):
        if len(args) == 0:
            return None

        if type(args[0]) is type:
            assert args[0] in (int, float)
            res, args = args[0] is int, args[1:]
        else:
            res = False

        assert 2 <= len(args) <= 3
        args = (args[1],) + args if len(args) == 2 else args
        return (res,) + tuple(tf.constant(i, tf.float32) for i in args)

    @classmethod
    def map(cls, floor, lo, center, hi, v):
        v, lo, hi = tf.math.abs(v), center, lo if v < 0 else hi
        if floor:
            v, lo, hi = tf.cond(
                lo < hi, lambda: (v, lo, hi), lambda: (1 - v, hi, lo))
            mapped = v * (hi - lo + 1) + lo - 0.5
            mapped = tf.cond(mapped == hi + 0.5, lambda: hi, lambda: tf.cond(
                mapped == lo - 0.5, lambda: lo, lambda: tf.math.round(mapped)))
            return tf.cast(mapped, tf.int32)
        else:
            return tf.cast(v * (hi - lo) + lo, tf.float32)

def normalize(*args, **kwargs):
    return Normalized((), (), *args, **kwargs)

class Augmentation:
    required, ops, track = False, (), ()
    def __new__(cls, *args, **kwargs):
        res, ops = super().__new__(cls), defaultdict(list)
        ops[cls].append((cls,))
        for i in cls.__mro__:
            parents = [j for j in i.__bases__ if __class__ in j.__mro__]
            base = [ops[i]] if len(parents) == 1 else [
                [(j, n) + ops[i][0][1:]] for n, j in enumerate(parents)]
            for p, b in zip(parents, base):
                ops[p] += b
        res.ops = OpsList(res, tuple(i[0] for i in sorted(
            ops[__class__], key=lambda x: x[::-1])), args, kwargs)
        return res

    def __init__(self, *args, **kwargs):
        pass

    def caller(self, cls, *args, **kwargs):
        return cls.call(self, *args, **kwargs)

    @property
    def variables(self):
        return tuple(getattr(*i) for i in self.ops.tracking)

    @variables.setter
    def variables(self, value):
        for i, j in zip(self.ops.tracking, value):
            setattr(*i, j)

class Group(Augmentation):
    required = True

    def call(self, im):
        return im

class Convert01(Augmentation):
    required = True

    def call(self, im):
        return im / 255

class Pipeline(Group):
    def __call__(self, im):
        for op in self.ops:
            im = op(im)
        return im

class Reformat(Augmentation):
    required = True

    def __init__(self, *args, data_format="channels_first", **kwargs):
        super().__init__(*args, **kwargs)
        self.channels_last = data_format == "channels_last"

    def call(self, im):
        im = im if self.channels_last else tf.transpose(im, [2, 0, 1])
        return im * 2 - 1
