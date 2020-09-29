from models.efficientnet.efficientnet import (
    Embedding as BaseEmbedding,
    Classifier as BaseClassifier,
)
from models.border.layers import Conv2D
from models.efficientnet.mbconv import MBConv as BaseMBConv
from models.efficientnet.train import Trainer as BaseTrainer

class MBConv(BaseMBConv):
    conv = Conv2D

class Embedding(BaseEmbedding):
    base = MBConv

class Classifier(BaseClassifier):
    base = MBConv

class Trainer(BaseTrainer):
    base = Classifier
