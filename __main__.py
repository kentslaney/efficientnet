import tensorflow as tf
import tensorflow_datasets as tfds
from efficientnet.border import BorderTrainer

model = BorderTrainer.from_preset(0, data_format="channels_last")
train, test = tfds.load('mnist', split=['train', 'test'], shuffle_files=True,
                        as_supervised=True)
train = train.map(lambda x, y: (tf.tile(x, (1, 1, 3)), y),
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = test.map(lambda x, y: (tf.tile(x, (1, 1, 3)), y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
model.fit(train, test)
