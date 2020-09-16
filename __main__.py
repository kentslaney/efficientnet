import tensorflow as tf
from preprocessing.randaugment import RandAugmentCrop
import matplotlib.pyplot as plt

im = tf.cast(tf.random.uniform((200, 200, 3)) * 256, tf.uint8)
im = RandAugmentCrop((100, 100))(im) / 2 + 0.5
plt.imshow(im)
plt.show()
