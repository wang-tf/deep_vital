from torch import nn
from mmengine.model import BaseModel
from mmcv.cnn import build_conv_layer
from deep_vital.registry import MODELS


@MODELS.register_module()
class LSTMBackbone(BaseModel):
    def __init__(self, in_channels, conv_cfg=dict(type='Conv1d'), init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.relu = nn.ReLU(inplace=True)

        self.conv = build_conv_layer(conv_cfg,
                                      in_channels,
                                      64,
                                      kernel_size=5,
                                      stride=1,
                                      padding=[4, 0],
                                      dilation=1,
                                      bias=True)

        self.layer1 = nn.LSTM(64, 128, bidirectional=True)
        self.layer2 = nn.LSTM(128, 128, bidirectional=True)
        self.layer3 = nn.LSTM(128, 64, bidirectional=True)

        self.layer4 = nn.Linear(64, 512)
        self.layer5 = nn.Linear(512, 256)
        self.layer6 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)
        x = self.relu(x)
        x = self.layer5(x)
        x = self.relu(x)
        x = self.layer6(x)
        x = self.relu(x)
        return x
