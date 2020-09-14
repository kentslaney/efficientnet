import tensorflow as tf
from preprocessing.randaugment import RandAugment
import matplotlib.pyplot as plt

im = tf.cast(tf.random.uniform((200, 200, 3)) * 256, tf.uint8)
im = RandAugment()(im)
plt.imshow(im)
plt.show()
