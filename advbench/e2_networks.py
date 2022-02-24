import sys
sys.setrecursionlimit(2000) # to allow the e2wrn28_10R model to be exported as a torch.nn.Module
import os.path
from typing import Tuple

import torch.nn.functional as F

from e2cnn import nn
from e2cnn import gspaces
from e2cnn.nn import init
import torch

import math
import numpy as np

STORE_PATH = "./models/stored/"

CHANNELS_CONSTANT = 1


def _get_fco(fco):
    if fco > 0.:
        fco *= np.pi
    return fco


def conv7x7(in_type: nn.FieldType, out_type: nn.FieldType, stride=1, padding=3, dilation=1, bias=False, sigma=None, F=1., initialize=True):
    """7x7 convolution with padding"""
    fco = _get_fco(F)
    return nn.R2Conv(in_type, out_type, 7,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     bias=bias,
                     sigma=sigma,
                     frequencies_cutoff=fco,
                     initialize=initialize
                     )


def conv5x5(in_type: nn.FieldType, out_type: nn.FieldType, stride=1, padding=2, dilation=1, bias=False, sigma=None, F=1., initialize=True):
    """5x5 convolution with padding"""
    fco = _get_fco(F)
    return nn.R2Conv(in_type, out_type, 5,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     bias=bias,
                     sigma=sigma,
                     frequencies_cutoff=fco,
                     initialize=initialize
                     )


def conv3x3(in_type: nn.FieldType, out_type: nn.FieldType, padding=1, stride=1, dilation=1, bias=False, sigma=None, F=1., initialize=True):
    """3x3 convolution with padding"""
    fco = _get_fco(F)
    return nn.R2Conv(in_type, out_type, 3,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     bias=bias,
                     sigma=sigma,
                     frequencies_cutoff=fco,
                     initialize=initialize
                     )


def conv1x1(in_type: nn.FieldType, out_type: nn.FieldType, padding=0, stride=1, dilation=1, bias=False, sigma=None, F=1., initialize=True):
    """1x1 convolution"""
    fco = _get_fco(F)
    return nn.R2Conv(in_type, out_type, 1,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     bias=bias,
                     sigma=sigma,
                     frequencies_cutoff=fco,
                     initialize=initialize
                     )


def regular_fiber(gspace: gspaces.GeneralOnR2, planes: int, fixparams: bool = True):
    """ build a regular fiber with the specified number of channels"""
    assert gspace.fibergroup.order() > 0
    N = gspace.fibergroup.order()
    planes = planes / N
    if fixparams:
        planes *= math.sqrt(N * CHANNELS_CONSTANT)
    planes = int(planes)
    
    return nn.FieldType(gspace, [gspace.regular_repr] * planes)


def quotient_fiber(gspace: gspaces.GeneralOnR2, planes: int, fixparams: bool = True):
    """ build a quotient fiber with the specified number of channels"""
    N = gspace.fibergroup.order()
    assert N > 0
    if isinstance(gspace, gspaces.FlipRot2dOnR2):
        n = N/2
        subgroups = []
        for axis in [0, round(n/4), round(n/2)]:
            subgroups.append((int(axis), 1))
    elif isinstance(gspace, gspaces.Rot2dOnR2):
        assert N % 4 == 0
        # subgroups = [int(round(N/2)), int(round(N/4))]
        subgroups = [2, 4]
    elif isinstance(gspace, gspaces.Flip2dOnR2):
        subgroups = [2]
    else:
        raise ValueError(f"Space {gspace} not supported")
    
    rs = [gspace.quotient_repr(subgroup) for subgroup in subgroups]
    size = sum([r.size for r in rs])
    planes = planes / size
    if fixparams:
        planes *= math.sqrt(N * CHANNELS_CONSTANT)
    planes = int(planes)
    return nn.FieldType(gspace, rs * planes).sorted()


def trivial_fiber(gspace: gspaces.GeneralOnR2, planes: int, fixparams: bool = True):
    """ build a trivial fiber with the specified number of channels"""

    if fixparams:
        planes *= math.sqrt(gspace.fibergroup.order() * CHANNELS_CONSTANT)
    planes = int(planes)
    return nn.FieldType(gspace, [gspace.trivial_repr] * planes)


def mixed_fiber(gspace: gspaces.GeneralOnR2, planes: int, ratio: float, fixparams: bool = True):

    N = gspace.fibergroup.order()
    assert N > 0
    if isinstance(gspace, gspaces.FlipRot2dOnR2):
        subgroup = (0, 1)
    elif isinstance(gspace, gspaces.Flip2dOnR2):
        subgroup = 1
    else:
        raise ValueError(f"Space {gspace} not supported")
    
    qr = gspace.quotient_repr(subgroup)
    rr = gspace.regular_repr
    
    planes = planes / rr.size
    
    if fixparams:
        planes *= math.sqrt(N * CHANNELS_CONSTANT)
    
    r_planes = int(planes * ratio)
    q_planes = int(2*planes * (1-ratio))
    
    return nn.FieldType(gspace, [rr] * r_planes + [qr] * q_planes).sorted()


def mixed1_fiber(gspace: gspaces.GeneralOnR2, planes: int, fixparams: bool = True):
    return mixed_fiber(gspace=gspace, planes=planes, ratio=0.5, fixparams=fixparams)


def mixed2_fiber(gspace: gspaces.GeneralOnR2, planes: int, fixparams: bool = True):
    return mixed_fiber(gspace=gspace, planes=planes, ratio=0.25, fixparams=fixparams)


FIBERS = {
    "trivial": trivial_fiber,
    "quotient": quotient_fiber,
    "regular": regular_fiber,
    "mixed1": mixed1_fiber,
    "mixed2": mixed2_fiber,
}


class WideBasic(nn.EquivariantModule):
    
    def __init__(self,
                 in_fiber: nn.FieldType,
                 inner_fiber: nn.FieldType,
                 dropout_rate, stride=1,
                 out_fiber: nn.FieldType = None,
                 F: float = 1.,
                 sigma: float = 0.45,
                 ):
        super(WideBasic, self).__init__()
        
        if out_fiber is None:
            out_fiber = in_fiber
        
        self.in_type = in_fiber
        inner_class = inner_fiber
        self.out_type = out_fiber
        
        if isinstance(in_fiber.gspace, gspaces.FlipRot2dOnR2):
            rotations = in_fiber.gspace.fibergroup.rotation_order
        elif isinstance(in_fiber.gspace, gspaces.Rot2dOnR2):
            rotations = in_fiber.gspace.fibergroup.order()
        else:
            rotations = 0
        
        if rotations in [0, 2, 4]:
            conv = conv3x3
        else:
            conv = conv5x5
        
        self.bn1 = nn.InnerBatchNorm(self.in_type)
        self.relu1 = nn.ReLU(self.in_type, inplace=True)
        self.conv1 = conv(self.in_type, inner_class, sigma=sigma, F=F, initialize=False)
        
        self.bn2 = nn.InnerBatchNorm(inner_class)
        self.relu2 = nn.ReLU(inner_class, inplace=True)
        
        self.dropout = nn.PointwiseDropout(inner_class, p=dropout_rate)
        
        self.conv2 = conv(inner_class, self.out_type, stride=stride, sigma=sigma, F=F, initialize=False)
        
        self.shortcut = None
        if stride != 1 or self.in_type != self.out_type:
            self.shortcut = conv1x1(self.in_type, self.out_type, stride=stride, bias=False, sigma=sigma, F=F, initialize=False)
            # if rotations in [0, 2, 4]:
            #     self.shortcut = conv1x1(self.in_type, self.out_type, stride=stride, bias=False, sigma=sigma, F=F, initialize=False)
            # else:
            #     self.shortcut = conv3x3(self.in_type, self.out_type, stride=stride, bias=False, sigma=sigma, F=F, initialize=False)
    
    def forward(self, x):
        x_n = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(x_n)))
        out = self.dropout(out)
        out = self.conv2(out)
        
        if self.shortcut is not None:
            out += self.shortcut(x_n)
        else:
            out += x
        
        return out
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape


class Wide_ResNet(torch.nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes=100,
                 N: int = 8,
                 r: int = 1,
                 f: bool = True,
                 main_fiber: str = "regular",
                 inner_fiber: str = "regular",
                 F: float = 1.,
                 sigma: float = 0.45,
                 deltaorth: bool = False,
                 fixparams: bool = True,
                 initial_stride: int = 1,
                 conv2triv: bool = True,
                 ):
        super(Wide_ResNet, self).__init__()
        
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor
        
        print(f'| Wide-Resnet {depth}x{k} ({CHANNELS_CONSTANT * 100}%)')
        
        nStages = [16, 16 * k, 32 * k, 64 * k]
        
        self.distributed = False
        self._fixparams = fixparams
        self.conv2triv = conv2triv
        
        self._layer = 0
        self._N = N
        
        # if the model is [F]lip equivariant
        self._f = f
        
        # level of [R]estriction:
        #   r < 0 : never do restriction, i.e. initial group (either D8 or C8) preserved for the whole network
        #   r = 0 : do restriction before first layer, i.e. initial group doesn't have rotation equivariance (C1 or D1)
        #   r > 0 : restrict after every block, i.e. start with 8 rotations, then restrict to 4 and finally 1
        self._r = r
        
        self._F = F
        self._sigma = sigma
        
        if self._f:
            self.gspace = gspaces.FlipRot2dOnR2(N)
        else:
            self.gspace = gspaces.Rot2dOnR2(N)
        
        if self._r == 0:
            id = (0, 1) if self._f else 1
            self.gspace, _, _ = self.gspace.restrict(id)
        
        r1 = nn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)
        self.in_type = r1
        
        # r2 = FIBERS[main_fiber](self.gspace, nStages[0], fixparams=self._fixparams)
        r2 = FIBERS[main_fiber](self.gspace, nStages[0], fixparams=True)
        self._in_type = r2
        
        self.conv1 = conv5x5(r1, r2, sigma=sigma, F=F, initialize=False)
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=initial_stride,
                                       main_fiber=main_fiber,
                                       inner_fiber=inner_fiber)
        if self._r > 0:
            id = (0, 4) if self._f else 4
            self.restrict1 = self._restrict_layer(id)
        else:
            self.restrict1 = lambda x: x
        
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2,
                                       main_fiber=main_fiber,
                                       inner_fiber=inner_fiber)
        if self._r > 1:
            id = (0, 1) if self._f else 1
            self.restrict2 = self._restrict_layer(id)
        else:
            self.restrict2 = lambda x: x
        
        if self.conv2triv:
            out_fiber = "trivial"
        else:
            out_fiber = None
            
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2,
                                       main_fiber=main_fiber,
                                       inner_fiber=inner_fiber,
                                       out_fiber=out_fiber
                                       )
        
        self.bn1 = nn.InnerBatchNorm(self.layer3.out_type, momentum=0.9)
        if self.conv2triv:
            self.relu = nn.ReLU(self.bn1.out_type, inplace=True)
        else:
            self.mp = nn.GroupPooling(self.layer3.out_type)
            self.relu = nn.ReLU(self.mp.out_type, inplace=True)
            
        self.linear = torch.nn.Linear(self.relu.out_type.size, num_classes)
        
        for name, module in self.named_modules():
            if isinstance(module, nn.R2Conv):
                if deltaorth:
                    init.deltaorthonormal_init(module.weights.data, module.basisexpansion)
                else:
                    init.generalized_he_init(module.weights.data, module.basisexpansion)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.Linear):
                module.bias.data.zero_()
        
        print("MODEL TOPOLOGY:")
        # for i, (name, mod) in enumerate(self.named_modules()):
        #     print(f"\t{i} - {name}")
        # for i, (name, mod) in enumerate(self.named_modules()):
        #     params = sum([p.numel() for p in mod.parameters() if p.requires_grad])
        #     if isinstance(mod, nn.EquivariantModule) and isinstance(mod.in_type, nn.FieldType) and isinstance(mod.out_type,
        #                                                                                                 nn.FieldType):
        #         print(f"\t{i: <3} - {name: <70} | {params: <8} | {mod.in_type.size: <4}- {mod.out_type.size: <4}")
        #     else:
        #         print(f"\t{i: <3} - {name: <70} | {params: <8} |")
        tot_param = sum([p.numel() for p in self.parameters() if p.requires_grad])
        print("Total number of parameters:", tot_param)
        self.exported=False

    def _restrict_layer(self, subgroup_id):
        layers = list()
        layers.append(nn.RestrictionModule(self._in_type, subgroup_id))
        layers.append(nn.DisentangleModule(layers[-1].out_type))
        self._in_type = layers[-1].out_type
        self.gspace = self._in_type.gspace
        
        restrict_layer = nn.SequentialModule(*layers)
        return restrict_layer
    
    def _wide_layer(self, block, planes: int, num_blocks: int, dropout_rate: float, stride: int,
                    main_fiber: str = "regular",
                    inner_fiber: str = "regular",
                    out_fiber: str = None,
                    ):
        
        self._layer += 1
        print("start building", self._layer)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        main_type = FIBERS[main_fiber](self.gspace, planes, fixparams=self._fixparams)
        inner_class = FIBERS[inner_fiber](self.gspace, planes, fixparams=self._fixparams)
        if out_fiber is None:
            out_fiber = main_fiber
        out_type = FIBERS[out_fiber](self.gspace, planes, fixparams=self._fixparams)
        
        for b, stride in enumerate(strides):
            if b == num_blocks - 1:
                out_f = out_type
            else:
                out_f = main_type
            layers.append(
                block(self._in_type, inner_class, dropout_rate, stride, out_fiber=out_f, sigma=self._sigma, F=self._F))
            self._in_type = out_f
        print("built", self._layer)
        return nn.SequentialModule(*layers)
    
    def features(self, x):
        
        x = nn.GeometricTensor(x, self.in_type)
        
        out = self.conv1(x)
        
        x1 = self.layer1(out)
        
        if self.distributed:
            x1.tensor = x1.tensor.cuda(1)
        
        x2 = self.layer2(self.restrict1(x1))
        
        if self.distributed:
            x2.tensor = x2.tensor.cuda(2)
        
        x3 = self.layer3(self.restrict2(x2))
        # out = self.relu(self.mp(self.bn1(out)))
        
        return x1, x2, x3

    def export(self):
        if not self.exported:
            self.exported=True
            self.export().eval()
    
    def unexport(self):
        if self.exported:
            self.exported=False
            self.unexport().train()
    
    def forward(self, x):
        if not self.exported:
            x = nn.GeometricTensor(x, self.in_type)
        
        out = self.conv1(x)
        out = self.layer1(out)
        
        if self.distributed:
            out.tensor = out.tensor.cuda(1)
        
        out = self.layer2(self.restrict1(out))
        
        if self.distributed:
            out.tensor = out.tensor.cuda(2)
        
        out = self.layer3(self.restrict2(out))
        
        if self.distributed:
            out.tensor = out.tensor.cuda(3)
        
        out = self.bn1(out)
        if not self.conv2triv:
            out = self.mp(out)
        out = self.relu(out)
        
        if not self.exported:
            out = out.tensor
        
        b, c, w, h = out.shape
        out = F.avg_pool2d(out, (w, h))
        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out


def e2wrn28_10R(**kwargs):
    """Constructs a Wide ResNet 28-10 model.
    This model is only [R]otation equivariant (no flips equivariance)
    Args:
        pretrained (bool): If True, returns a model pre-trained on Cifar100
    """
    model = Wide_ResNet(28, 7, 0.3, f=False, initial_stride=1, **kwargs)
    return model

def wrn28_10(**kwargs):
    """Constructs a Wide ResNet 28-10 model.
    This model is only [R]otation equivariant (no flips equivariance)
    Args:
        pretrained (bool): If True, returns a model pre-trained on Cifar100
    """
    model = Wide_ResNet(28, 7, 0.3, f=False, initial_stride=1, **kwargs)
    model.export()
    return model
