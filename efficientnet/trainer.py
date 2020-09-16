import tensorflow as tf
from preprocessing.randaugment import RandAugmentCrop, PrepStretch
from .efficientnet import Classifier

presets = [ # size, *args
    (224, 1.0, 1.0, 0.2), # B0
    (240, 1.0, 1.1, 0.2), # B1
    (260, 1.1, 1.2, 0.3), # B2
    (300, 1.2, 1.4, 0.3), # B3
    (380, 1.4, 1.8, 0.4), # B4
    (456, 1.6, 2.2, 0.4), # B5
    (528, 1.8, 2.6, 0.5), # B6
    (600, 2.0, 3.1, 0.5), # B7
    (800, 4.3, 5.3, 0.5), # L2
]

class Trainer(Classifier):
    def __init__(self, size, *args, data_format="channels_first", **kwargs):
        super().__init__(*args, data_format=data_format, **kwargs)
        self._aug = RandAugmentCrop((size, size), data_format=data_format)
        self._prep = PrepStretch((size, size), data_format=data_format)
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
        x = x.shuffle(1000)
        x = x.batch(batch_size)
        x = x.prefetch(tf.data.experimental.AUTOTUNE)
        if validation_data is not None:
            validation_data = validation_data.map(
                self.prep, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            validation_data = validation_data.batch(validation_batch_size)
            validation_data = validation_data.prefetch(
                tf.data.experimental.AUTOTUNE)
        return super().fit(x, validation_data=validation_data, **kwargs)

    @classmethod
    def from_preset(cls, i, **kwargs):
        return cls(*presets[i], **kwargs)
