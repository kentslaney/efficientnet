import tensorflow as tf
from inspect import signature
from .base import Reformat, Convert01, Group, Pipeline
from .adjust import Adjust
from .transform import PaddedTransforms, CroppedTransforms, Stretch, CenterCrop

class Randomize(Group):
    def __init__(self, *args, n=3, m=.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.n, self.m = self.ops.choosable if n == -1 else n, m

    def __call__(self, im):
        for i, op in enumerate(self.ops.sample(self.n)):
            inputs = len(signature(op).parameters) - 1
            m = self.m * tf.math.sign(tf.random.uniform((inputs,)) - 0.5)
            im = op(im, *tf.unstack(m))
        return im

class RandAugmentPadded(Randomize):
    ops = (Adjust, PaddedTransforms, Reformat)

class RandAugmentCropped(Randomize):
    ops = (Adjust, CroppedTransforms, Reformat)

class PrepStretched(Pipeline):
    ops = (Convert01, Stretch, Reformat)

class PrepCropped(Pipeline):
    ops = (Convert01, CenterCrop, Reformat)
