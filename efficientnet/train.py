import tensorflow as tf
from .efficientnet import Classifier
from train import RandAugmentTrainer, TFDSTrainer
from utils import RequiredLength

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

class Trainer(RandAugmentTrainer, TFDSTrainer):
    base = Classifier

    def build(self, size=None, preset=0, custom=None, name=None, **kwargs):
        if custom is None:
            _name, _size, *custom = presets[preset]
            size = _size if size is None else size
            name = _name if name is None else name
        super().build(size=size, **kwargs)

        self.model = self.base(*custom, name=name, outputs=self.outputs,
                               data_format=self.data_format)
        self.compile("sparse_categorical_crossentropy",
                     ["sparse_categorical_accuracy"])

    def cli(self, parser):
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument("--preset", metavar="N", type=int, default=0, help=(
            "which preset to use; 0-7 correspond to B0 to B7, and 8 is L2"))
        group.add_argument(
            "--custom", type=float, default=None, action=RequiredLength(2, 5),
            metavar=("WIDTH", "DEPTH", "DROPOUT", "DIVISOR", "STEM"),
            help="use a custom initialization")
        super().cli(parser)
