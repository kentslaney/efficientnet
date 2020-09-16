import tensorflow as tf
from preprocessing.randaugment import RandAugmentCrop, PrepStretch
from .efficientnet import Classifier

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

class Trainer(Classifier):
    def __init__(self, size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aug = RandAugmentCrop((size, size), data_format=self.data_format)
        self._prep = PrepStretch((size, size), data_format=self.data_format)
        self.aug = lambda x, *y: (self._aug(x),) + y
        self.prep = lambda x, *y: (self._prep(x),) + y
        self.compile("adam", "sparse_categorical_crossentropy",
                     ["sparse_categorical_accuracy"])

    def fit(self, x, validation_data=None, batch_size=None,
            validation_batch_size=None, **kwargs):
        batch_size = 32 if batch_size is None else batch_size
        validation_batch_size = batch_size if validation_batch_size is None \
            else validation_batch_size
        x = x.map(self.aug, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        x = x.shuffle(1000).batch(batch_size).prefetch(
            tf.data.experimental.AUTOTUNE)
        if validation_data is not None:
            validation_data = validation_data.map(
                self.prep, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            validation_data = validation_data.batch(validation_batch_size)
            validation_data = validation_data.prefetch(
                tf.data.experimental.AUTOTUNE)
        return super().fit(x, validation_data=validation_data, **kwargs)

    @classmethod
    def from_preset(cls, i, **kwargs):
        name, *args = presets[i]
        return cls(*args, name=name, **kwargs)
