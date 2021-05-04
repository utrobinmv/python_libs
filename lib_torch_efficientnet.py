#!pip install efficientnet_pytorch

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b2')