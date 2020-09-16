from border.layers import Conv2D
from .mbconv import MBConv
from .trainer import Trainer

class MBBorderConv(MBConv):
    conv = Conv2D

class BorderTrainer(Trainer):
    base = MBConv
    conv = Conv2D
