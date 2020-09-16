from .layers import Conv2D
from efficientnet.mbconv import MBConv as BaseMBConv
from efficientnet.trainer import Trainer as BaseTrainer

class MBConv(BaseMBConv):
    conv = Conv2D

class Trainer(BaseTrainer):
    base = MBConv
    conv = Conv2D
