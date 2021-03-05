from models.efficientnet.efficientnet import (
    Embedding as BaseEmbedding,
    Classifier as BaseClassifier,
)
from .layers import Conv2D, DepthwiseConv2D
from models.efficientnet.mbconv import MBConv as BaseMBConv
from models.efficientnet.train import Trainer as BaseTrainer

class MBConv(BaseMBConv):
    depthwise = DepthwiseConv2D
    conv = Conv2D

class Embedding(BaseEmbedding):
    base = MBConv

class Classifier(BaseClassifier):
    base = MBConv

class Trainer(BaseTrainer):
    base = Classifier
