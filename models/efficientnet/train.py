# Copyright (C) 2020 by Kent Slaney <kent@slaney.org>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

import tensorflow as tf
from tensorflow_addons.optimizers import MovingAverage
from .efficientnet import Classifier
from models.train import RandAugmentTrainer, TFDSTrainer
from cli.utils import RequiredLength
from models.utils import TPUBatchNormalization

presets = [
    ("EfficientNet-B0", 224, 1.0, 1.0, 0.2),
    ("EfficientNet-B1", 240, 1.0, 1.1, 0.2),
    ("EfficientNet-B2", 260, 1.1, 1.2, 0.3),
    ("EfficientNet-B3", 300, 1.2, 1.4, 0.3),
    ("EfficientNet-B4", 380, 1.4, 1.8, 0.4),
    ("EfficientNet-B5", 456, 1.6, 2.2, 0.4),
    ("EfficientNet-B6", 528, 1.8, 2.6, 0.5),
    ("EfficientNet-B7", 600, 2.0, 3.1, 0.5),
    ("EfficientNet-L2", 800, 4.3, 5.3, 0.5),
]

tf.config.optimizer.set_jit(True)

class Trainer(RandAugmentTrainer, TFDSTrainer):
    base = Classifier
    opt = lambda _, lr: MovingAverage(
        tf.keras.optimizers.RMSprop(lr, 0.9, 0.9, 0.001))

    def build(self, size=None, preset=0, custom=None, name=None, **kwargs):
        if custom is None:
            _name, _size, *custom = presets[preset]
            size = _size if size is None else size
            name = _name if name is None else name
        super().build(size=size, **kwargs)
        self.mapper = lambda f: lambda x, y: (
            f(x), tf.one_hot(y, self.outputs))

        if self.tpu:
            self.base.base.bn = TPUBatchNorm
        self.model = self.base(
            *custom, name=name, outputs=self.outputs, pretranspose=self.tpu,
            data_format=self.data_format)

        self.compile(tf.keras.losses.CategoricalCrossentropy(True, 0.1),
                     ["categorical_accuracy"])

    @classmethod
    def cli(cls, parser):
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument("--preset", metavar="N", type=int, default=0, help=(
            "which preset to use; 0-7 correspond to B0 to B7, and 8 is L2"))
        group.add_argument(
            "--custom", type=float, default=None, action=RequiredLength(2, 5),
            metavar=("WIDTH", "DEPTH", "DROPOUT", "DIVISOR", "STEM"),
            help="use a custom initialization")
        super().cli(parser)
