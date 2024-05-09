import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
from collections import OrderedDict
from torch import Tensor
from functools import partial
from typing import Any, List, Optional, Tuple
import torch.utils.checkpoint as cp
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
#from models.route import *
__all__ = [
    "DenseNet",
    "DenseNet121_Weights",
    "DenseNet161_Weights",
    "DenseNet169_Weights",
    "DenseNet201_Weights",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
]
class L2pooling(nn.Module): # 实现了L2池化操作。具体功能是对输入进行平方运算，然后利用预定的高斯核进行卷积，最后开方得到输出
    def __init__(self, filter_size=5, stride=1, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        # 构造一个高斯核函数
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()
    
class ChannelAttn(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)
        self.act_layer = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _x = x
        x = self.gap(x)
        x = self.conv2(self.act_layer(self.conv1(x)))
        x = self.sigmoid(x)
        x = _x * x
        return x


class DAB(nn.Module):
    def __init__(self, embed_dim, dab_layers):
        super().__init__()
        #self.conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        channel_attn = []
        for i in range(dab_layers):
            channel_attn += [ChannelAttn(embed_dim=embed_dim)]
        self.channel_attn = nn.Sequential(*channel_attn)
    def forward(self, x):
        x = self.channel_attn(x)
        return x
    
class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:  # noqa: F811
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        memory_efficient: bool = False,
    ) -> None:

        super().__init__()
        _log_api_usage_once(self)

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        # self.l1 = L2pooling(channels=128)
        # self.l2 = L2pooling(channels=256)
        # self.l3 = L2pooling(channels=512)
        # self.l4 = L2pooling(channels=1024)
        # self.drop2d = nn.Dropout(p=0.1)
        # self.averagepool = nn.AdaptiveAvgPool2d(8)
        self.insert1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.insert2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)

        self.insert3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.insert4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)

        self.insert5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.insert6 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)

        # self.insert7 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False)
        # self.insert8 = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False)

        self.dab1 = DAB(embed_dim=128, dab_layers=1)
        self.dab2 = DAB(embed_dim=256, dab_layers=1)
        self.dab3 = DAB(embed_dim=512, dab_layers=2)
        
        #self.conv =nn.Conv2d(1920, 1024, kernel_size=1, stride=1)

        # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):                
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:

        for name in list(self.features.children()):
            x = name(x)
            
            if isinstance(name, _Transition) and x.size()[1]==128:
           
                insert = self.dab1(x)
                insert= self.insert2(self.insert1(insert))

            if isinstance(name, _Transition) and x.size()[1]==256:
                insert = self.dab2(x) + insert
                insert = self.insert4(self.insert3(insert))

            if isinstance(name, _Transition) and x.size()[1]==512:
                insert = self.dab3(x) + insert
                insert = self.insert6(insert)

        # out = self.drop2d(self.l4(F.normalize(x, dim=1, p=2)))
        # out = torch.cat([out1, out2, out3, out], dim=1)
        # out = self.conv(out)
        
        return x + insert
        
        # features = self.features(x)
        # out = F.relu(features, inplace=True)
        # out = F.adaptive_avg_pool2d(out, (1, 1))
        # out = torch.flatten(out, 1)
        # out = self.classifier(out)
        # return out


def _load_state_dict(model: nn.Module, weights: WeightsEnum, progress: bool) -> None:
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = weights.get_state_dict(progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)


def _densenet(
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> DenseNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)

    if weights is not None:
        _load_state_dict(model=model, weights=weights, progress=progress)

    return model


_COMMON_META = {
    "min_size": (29, 29),
    "categories": _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/pull/116",
    "_docs": """These weights are ported from LuaTorch.""",
}


class DenseNet121_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/densenet121-a639ec97.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 7978856,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 74.434,
                    "acc@5": 91.972,
                }
            },
            "_ops": 2.834,
            "_file_size": 30.845,
        },
    )
    DEFAULT = IMAGENET1K_V1


@handle_legacy_interface(weights=("pretrained", DenseNet121_Weights.IMAGENET1K_V1))
def densenet121(*, weights: Optional[DenseNet121_Weights] = None, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-121 model from
    `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.

    Args:
        weights (:class:`~torchvision.models.DenseNet121_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.DenseNet121_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.densenet.DenseNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.DenseNet121_Weights
        :members:
    """
    weights = DenseNet121_Weights.verify(weights)

    return _densenet(32, (6, 12, 24, 16), 64, weights, progress, **kwargs)

# The dictionary below is internal implementation detail and will be removed in v0.15
from torchvision.models._utils import _ModelURLs


model_urls = _ModelURLs(
    {
        "densenet121": DenseNet121_Weights.IMAGENET1K_V1.url,
    }
)

