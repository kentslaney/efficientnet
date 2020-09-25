import efficientnet.efficientnet
from border.layers import Conv2D
from efficientnet.mbconv import MBConv as BaseMBConv
from efficientnet.train import Trainer as BaseTrainer

class MBConv(BaseMBConv):
    conv = Conv2D

class Embedding(efficientnet.efficientnet.Embedding):
    base = MBConv

class Classifier(efficientnet.efficientnet.Classifier):
    base = MBConv

class Trainer(BaseTrainer):
    base = Classifier
