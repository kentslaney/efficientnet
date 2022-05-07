import tensorflow as tf
from tensorflow_addons.optimizers import MovingAverage
from src.efficientnet import Classifier, Embedding
from src.base import RandAugmentTrainer, TFDSTrainer
from src.utils import TPUBatchNormalization, RequiredLength, cli_builder
from src.border import Conv2D, DepthwiseConv2D
from src.mbconv import MBConv
from src.simple import SimpleTrainer

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

    def opt(self, lr):
        return MovingAverage(tf.keras.optimizers.RMSprop(lr, 0.9, 0.9, 0.001))

    @cli_builder
    def __init__(self, dataset="imagenet2012", size=None, preset=0,
                 custom=None, name=None, **kw):
        if custom is None:
            _name, _size, *custom = presets[preset]
            size = _size if size is None else size
            name = _name if name is None else name
        super().__init__(dataset=dataset, size=size, **kw)

        self.model = self.base(
            *custom, outputs=self.outputs, pretranspose=self.tpu,
            data_format=self.data_format, name=name)
        if self.tpu:
            self.tpu_build()

        self.compile(tf.keras.losses.CategoricalCrossentropy(False, 0.1),
                     ["categorical_accuracy"])

    def tpu_build(self):
        self.model.base.bn = TPUBatchNorm

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
