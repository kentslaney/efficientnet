import tensorflow as tf
from tensorflow_addons.optimizers import MovingAverage
from efficientnet import Classifier, Embedding
from base import RandAugmentTrainer, TFDSTrainer
from utils import TPUBatchNormalization, RequiredLength, cli_builder
from border import Conv2D, DepthwiseConv2D
from mbconv import MBConv
from simple import SimpleTrainer

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


class EfficientnetTrainer(RandAugmentTrainer, TFDSTrainer):
    base = Classifier
    opt = lambda _, lr: MovingAverage(
        tf.keras.optimizers.RMSprop(lr, 0.9, 0.9, 0.001))

    @cli_builder
    def build(self, size=None, preset=0, custom=None, name=None, **kwargs):
        if custom is None:
            _name, _size, *custom = presets[preset]
            size = _size if size is None else size
            name = _name if name is None else name
        self.mapper = lambda f: lambda x, y: (
            f(x), tf.one_hot(y, self.outputs))
        super().build(size=size, **kwargs)

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
        parser.add_argument("--name", help=(
            "name to assign to the model; potentially useful to differentiate "
            "exports"))
        super().cli(parser)

class BorderMBConv(MBConv):
    # depthwise = DepthwiseConv2D
    conv = Conv2D

class BorderClassifier(Classifier):
    base = BorderMBConv

class BorderTrainer(EfficientnetTrainer):
    base = BorderClassifier

cli_names = {
    "base": EfficientnetTrainer,
    "border": BorderTrainer,
    "simple": SimpleTrainer,
}
