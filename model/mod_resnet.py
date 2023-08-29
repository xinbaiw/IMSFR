import torch
import torch.nn as nn
from torch.utils import model_zoo
from collections import OrderedDict
import math
# from .splat import SplAtConv2d, DropBlock2D




def load_weights_sequential(target, source_state, extra_chan=1):
    
    new_dict = OrderedDict()

    for k1, v1 in target.state_dict().items():
        if not 'num_batches_tracked' in k1:
            if k1 in source_state:
                tar_v = source_state[k1]

                if v1.shape != tar_v.shape:
                    # Init the new segmentation channel with zeros
                    # print(v1.shape, tar_v.shape)
                    c, _, w, h = v1.shape
                    pads = torch.zeros((c,extra_chan,w,h), device=tar_v.device)
                    nn.init.orthogonal_(pads)
                    tar_v = torch.cat([tar_v, pads], 1)

                new_dict[k1] = tar_v

    target.load_state_dict(new_dict, strict=False)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnest101': 'https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest101-22405ba7.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3), extra_chan=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3+extra_chan, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

def resnet18(pretrained=True, extra_chan=0):
    model = ResNet(BasicBlock, [2, 2, 2, 2], extra_chan)
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet18']), extra_chan)
    return model

def resnet50(pretrained=True, extra_chan=0):
    model = ResNet(Bottleneck, [3, 4, 6, 3], extra_chan)
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet50']), extra_chan)
    return model
def resnest101(pretrained=True, root='~/.encoding/models', extra_chan=0, **kwargs):
    model = ResNetaot(Bottleneckaot, [3, 4, 23, 3],
                   radix=2,
                   groups=1,
                   bottleneck_width=64,
                   deep_stem=True,
                   stem_width=64,
                   avg_down=True,
                   avd=True,
                   avd_first=False,
                   **kwargs)
    # if pretrained:
    #     model.load_state_dict(
    #         torch.hub.load_state_dict_from_url(
    #             resnest_model_urls['resnest101'],
    #             progress=True,
    #             check_hash=True))

    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnest101']), extra_chan)
    return model

    # return model


class ResNetaot(nn.Module):
    """ResNet Variants
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    # pylint: disable=unused-variable
    def __init__(self,
                 block,
                 layers,
                 radix=1,
                 groups=1,
                 bottleneck_width=64,
                 num_classes=1000,
                 dilated=False,
                 dilation=1,
                 deep_stem=False,
                 stem_width=64,
                 avg_down=False,
                 rectified_conv=False,
                 rectify_avg=False,
                 avd=False,
                 avd_first=False,
                 final_drop=0.0,
                 dropblock_prob=0,
                 last_gamma=False,
                 norm_layer=nn.BatchNorm2d,
                 freeze_at=0):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        super(ResNetaot, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            from rfconv import RFConv2d
            conv_layer = RFConv2d
        else:
            conv_layer = nn.Conv2d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(3,
                           stem_width,
                           kernel_size=3,
                           stride=2,
                           padding=1,
                           bias=False,
                           **conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width,
                           stem_width,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bias=False,
                           **conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width,
                           stem_width * 2,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bias=False,
                           **conv_kwargs),
            )
        else:
            self.conv1 = conv_layer(3,
                                    64,
                                    kernel_size=7,
                                    stride=2,
                                    padding=3,
                                    bias=False,
                                    **conv_kwargs)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,
                                       64,
                                       layers[0],
                                       norm_layer=norm_layer,
                                       is_first=False)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block,
                                           256,
                                           layers[2],
                                           stride=1,
                                           dilation=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        elif dilation == 2:
            self.layer3 = self._make_layer(block,
                                           256,
                                           layers[2],
                                           stride=2,
                                           dilation=1,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block,
                                           256,
                                           layers[2],
                                           stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)

        self.stem = [self.conv1, self.bn1]
        self.stages = [self.layer1, self.layer2, self.layer3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.freeze(freeze_at)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    norm_layer=None,
                    dropblock_prob=0.0,
                    is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(
                        nn.AvgPool2d(kernel_size=stride,
                                     stride=stride,
                                     ceil_mode=True,
                                     count_include_pad=False))
                else:
                    down_layers.append(
                        nn.AvgPool2d(kernel_size=1,
                                     stride=1,
                                     ceil_mode=True,
                                     count_include_pad=False))
                down_layers.append(
                    nn.Conv2d(self.inplanes,
                              planes * block.expansion,
                              kernel_size=1,
                              stride=1,
                              bias=False))
            else:
                down_layers.append(
                    nn.Conv2d(self.inplanes,
                              planes * block.expansion,
                              kernel_size=1,
                              stride=stride,
                              bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(
                block(self.inplanes,
                      planes,
                      stride,
                      downsample=downsample,
                      radix=self.radix,
                      cardinality=self.cardinality,
                      bottleneck_width=self.bottleneck_width,
                      avd=self.avd,
                      avd_first=self.avd_first,
                      dilation=1,
                      is_first=is_first,
                      rectified_conv=self.rectified_conv,
                      rectify_avg=self.rectify_avg,
                      norm_layer=norm_layer,
                      dropblock_prob=dropblock_prob,
                      last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(
                block(self.inplanes,
                      planes,
                      stride,
                      downsample=downsample,
                      radix=self.radix,
                      cardinality=self.cardinality,
                      bottleneck_width=self.bottleneck_width,
                      avd=self.avd,
                      avd_first=self.avd_first,
                      dilation=2,
                      is_first=is_first,
                      rectified_conv=self.rectified_conv,
                      rectify_avg=self.rectify_avg,
                      norm_layer=norm_layer,
                      dropblock_prob=dropblock_prob,
                      last_gamma=self.last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      radix=self.radix,
                      cardinality=self.cardinality,
                      bottleneck_width=self.bottleneck_width,
                      avd=self.avd,
                      avd_first=self.avd_first,
                      dilation=dilation,
                      rectified_conv=self.rectified_conv,
                      rectify_avg=self.rectify_avg,
                      norm_layer=norm_layer,
                      dropblock_prob=dropblock_prob,
                      last_gamma=self.last_gamma))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        xs = []

        x = self.layer1(x)
        xs.append(x)  # 4X
        x = self.layer2(x)
        xs.append(x)  # 8X
        x = self.layer3(x)
        xs.append(x)  # 16X
        # Following STMVOS, we drop stage 5.
        xs.append(x)  # 16X

        return xs

    def freeze(self, freeze_at):
        if freeze_at >= 1:
            for m in self.stem:
                freeze_params(m)

        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                freeze_params(stage)


def freeze_params(module):
    for p in module.parameters():
        p.requires_grad = False

class Bottleneckaot(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 radix=1,
                 cardinality=1,
                 bottleneck_width=64,
                 avd=False,
                 avd_first=False,
                 dilation=1,
                 is_first=False,
                 rectified_conv=False,
                 rectify_avg=False,
                 norm_layer=None,
                 dropblock_prob=0.0,
                 last_gamma=False):
        super(Bottleneckaot, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes,
                               group_width,
                               kernel_size=1,
                               bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplAtConv2d(group_width,
                                     group_width,
                                     kernel_size=3,
                                     stride=stride,
                                     padding=dilation,
                                     dilation=dilation,
                                     groups=cardinality,
                                     bias=False,
                                     radix=radix,
                                     rectify=rectified_conv,
                                     rectify_avg=rectify_avg,
                                     norm_layer=norm_layer,
                                     dropblock_prob=dropblock_prob)
        elif rectified_conv:
            from rfconv import RFConv2d
            self.conv2 = RFConv2d(group_width,
                                  group_width,
                                  kernel_size=3,
                                  stride=stride,
                                  padding=dilation,
                                  dilation=dilation,
                                  groups=cardinality,
                                  bias=False,
                                  average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv2d(group_width,
                                   group_width,
                                   kernel_size=3,
                                   stride=stride,
                                   padding=dilation,
                                   dilation=dilation,
                                   groups=cardinality,
                                   bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(group_width,
                               planes * 4,
                               kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(planes * 4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
