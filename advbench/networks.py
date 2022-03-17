from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces
from e2cnn import nn as enn
from advbench.e2_utils import *
from torch.nn.functional import pad
from advbench.e2_networks import e2wrn28_10R
from advbench.wrn import Wide_ResNet
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

def Classifier(input_shape, num_classes, hparams):

    if input_shape[0] == 1:
        if hparams["model"] == "CnSteerableCNN":
            return CnSteerableCNN(num_channels=1)
        else:
            print(f"input shape {input_shape}, num classes {num_classes}")
            return MNISTNet(input_shape, num_classes)#CnSteerableCNN(num_classes)
    elif input_shape[0] == 3:
        # return models.resnet18(num_classes=num_classes)
        if hparams["model"] == "resnet18":
            return ResNet18(num_classes = num_classes)
        elif hparams["model"] == "steerable_resnet18":
            return SteerableResNet18(num_classes = num_classes)
        elif hparams["model"] == "wrn-28-10-rot":
            print("Using e2 invariant WRN-28-10")
            return e2wrn28_10R(num_classes=num_classes)
        elif hparams["model"] == "wrn-28-10":
            print("Using WRN-28-10")
            return wrn28_10(num_classes=num_classes)
        elif hparams["model"] == "CnSteerableCNN":
            return CnSteerableCNN(num_channels=3, num_classes = num_classes)
        else:
            raise Exception("Unknown model: {}".format(hparams["model"]))
    else:
        assert False


class MNISTNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)    #TODO(AR): might need to remove softmax for KL div in TRADES

"""Resnet implementation is based on the implementation found in:
https://github.com/YisenWang/MART/blob/master/resnet.py
"""

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3],num_classes=num_classes)


class CnSteerableCNN(torch.nn.Module):
    
    def __init__(self, n_classes=10, n_rot = 8, num_channels = 1):
        
        super(CnSteerableCNN, self).__init__()
        self.exported=False
        
        # the model is equivariant under rotations by 360/n_rot degrees, modelled by Cn_rot
        self.r2_act = gspaces.Rot2dOnR2(N=n_rot)
        
        # the input image is a scalars field, corresponding to the trivial representation
        in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr]*num_channels)
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C8
        out_type_1 = enn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.block1 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type_1, kernel_size=7, padding=2, bias=False),
            enn.InnerBatchNorm(out_type_1),
            enn.ReLU(out_type_1, inplace=True)
        )
        in_type_2 = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type_2 = enn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block2 = enn.SequentialModule(
            enn.R2Conv(in_type_2, out_type_2, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(out_type_2),
            enn.ReLU(out_type_2, inplace=True)
        )
        #self.pool1 = enn.SequentialModule(
        #    enn.PointwiseAvgPoolAntialiased(out_type_2, sigma=0.66, stride=2)
        #)
        self.pool1 = enn.PointwiseAdaptiveAvgPool(out_type_2, (14, 14))
        # convolution 3
        # the old output type is the input type to the next layer
        in_type_3 = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type_3 = enn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block3 = enn.SequentialModule(
            enn.R2Conv(in_type_3, out_type_3, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(out_type_3),
            enn.ReLU(out_type_3, inplace=True)
        )
        # convolution 4
        # the old output type is the input type to the next layer
        in_type_4 = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type_4 = enn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block4 = enn.SequentialModule(
            enn.R2Conv(in_type_4, out_type_4, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(out_type_4),
            enn.ReLU(out_type_4, inplace=True)
        )
        #self.pool2 = enn.SequentialModule(
        #    enn.PointwiseAvgPoolAntialiased(out_type_4, sigma=0.66, stride=2)
        #)
        self.pool2 = enn.PointwiseAdaptiveAvgPool(out_type_4, (7,7))
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type_5 = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type_5 = enn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block5 = enn.SequentialModule(
            enn.R2Conv(in_type_5, out_type_5, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(out_type_5),
            enn.ReLU(out_type_5, inplace=True)
        )
        
        # convolution 6
        # the old output type is the input type to the next layer
        in_type_6 = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type_6 = enn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block6 = enn.SequentialModule(
            enn.R2Conv(in_type_6, out_type_6, kernel_size=5, padding=1, bias=False),
            enn.InnerBatchNorm(out_type_6),
            enn.ReLU(out_type_6, inplace=True)
        )
        #self.pool3 = enn.PointwiseAvgPoolAntialiased(out_type_6, sigma=0.66, stride=1, padding=0)
        self.pool3 = enn.PointwiseAdaptiveAvgPool(out_type_6, (7,7))
        self.gpool = enn.GroupPooling(out_type_6)
        
        # number of output channels
        #c = self.gpool.out_type.size
        
        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(3136, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, n_classes),
        )
        self.model = enn.SequentialModule(self.block1,
        self.block2, self.pool1, self.block3, self.block4,
        self.pool2, self.block5, self.block6, self.pool3,
        self.gpool)
    
    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        if not self.exported:
            #x = pad(input, (0,1,0,1))
            x = enn.GeometricTensor(input, self.input_type)
        else:
            x = input
        '''
        #x = input
        # apply each equivariant block
        
        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        
        x = self.block5(x)
        x = self.block6(x)
        
        # pool over the spatial dimensions
        x = self.pool3(x)
        
        # pool over the group
        x = self.gpool(x)
        '''
        if not self.exported:
            x = self.model(x)
            # unwrap the output GeometricTensor
            # (take the Pytorch tensor and discard the associated representation)
            x = x.tensor
        else:
            x = self.exported_model(x)
        
        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
        return x

    def export(self):
        self.exported=True
        self.exported_model = self.model.export().eval()
    
    def unexport(self):
        self.exported=False

class SteerableResNet(nn.Module):
    def __init__(self, block, num_blocks, restriction = 1, flip=False, N=8,  main_fiber: str = "regular",
                 inner_fiber: str = "regular", num_classes=10):
        super(SteerableResNet, self).__init__()
        self.exported=False
        nStages = [64, 128, 256, 512]
        self.in_planes = 64
        # level of [R]estriction:
        #   restriction < 0 : never do restriction, i.e. initial group (either D8 or C8) preserved for the whole network
        #   restriction = 0 : do restriction before first layer, i.e. initial group doesn't have rotation equivariance (C1 or D1)
        #   restriction > 0 : restrict after every block, i.e. start with 8 rotations, then restrict to 4 and finally 1
        self.r = restriction
        if self.r < 0:
            nStages = [int(n//8) for n in nStages]
        elif self.r == 0:
            nStages[1:] = [int(n//8) for n in nStages[1:]]
        elif self.r > 0:
            for i in range(len(nStages)):
                nStages[i] = int(nStages[i]//(8/2**i))
        self.f = flip
        self.N = N
        if self.f:
            self.gspace = gspaces.FlipRot2dOnR2(N)
        else:
            self.gspace = gspaces.Rot2dOnR2(N)
        if self.r == 0:
            id = (0, 1) if self.f else 1
            self.gspace, _, _ = self.gspace.restrict(id)
        r1 = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)
        self.input_type = r1
        r2 = FIBERS[main_fiber](self.gspace, nStages[0], fixparams=True)
        self.in_type = r2
        self.in_planes = r2
        self.conv1 = enn.R2Conv(r1, r2, nStages[0], 3, stride=1, bias=False)
        self.relu1 = enn.ReLU(self.in_type,inplace=True)
        self.bn1 = enn.InnerBatchNorm(self.in_type)
        self.restrict1 = lambda x: x
        self.layer1 = self._make_layer_rot(block, nStages[0], num_blocks[0], stride=1, main_fiber=main_fiber, inner_fiber=inner_fiber)
        if self.r > 0:
            id = (0, 4) if self.f else 4
            self.restrict2 = self._restrict_layer(id)
        else:
            self.restrict2 = lambda x: x
        self.layer2 = self._make_layer_rot(block, nStages[1], num_blocks[1], stride=2, main_fiber=main_fiber, inner_fiber=inner_fiber)
        if self.r > 0:
            id = (0, 2) if self.f else 2
            self.restrict3 = self._restrict_layer(id)
        else:
            self.restrict3 = lambda x: x
        self.layer3 = self._make_layer_rot(block, nStages[2], num_blocks[2], stride=2, main_fiber=main_fiber, inner_fiber=inner_fiber)
        if self.r > 0:
            id = (0, 1) if self.f else 1
            self.restrict4 = self._restrict_layer(id)
        else:
            self.restrict4 = lambda x: x
        self.layer4 = self._make_layer_rot(block, nStages[3], num_blocks[3], stride=2, main_fiber=main_fiber, inner_fiber=inner_fiber)
        self.bn4 = enn.InnerBatchNorm(self.in_type, momentum=0.9)
        self.linear = nn.Linear(self.in_type.size, num_classes)

        self.total_params = sum(p.numel() for p in self.parameters())
        #print("Total number of parameters: {}".format(self.total_params))
        self.total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #print("Total number of trainable parameters: {}".format(self.total_trainable_params))

        for name, module in self.named_modules():
            if isinstance(module, enn.R2Conv):
                init.generalized_he_init(module.weights.data, module.basisexpansion)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.Linear):
                module.bias.data.zero_()

    def _restrict_layer(self, subgroup_id):
        layers = list()
        layers.append(enn.RestrictionModule(self.in_type, subgroup_id))
        layers.append(enn.DisentangleModule(layers[-1].out_type))
        self.in_type = layers[-1].out_type
        self.gspace = self.in_type.gspace
        restrict_layer = enn.SequentialModule(*layers)
        return restrict_layer

    def _make_layer_rot(self, block, planes, num_blocks, stride, main_fiber: str = "regular",
                    inner_fiber: str = "regular",
                    out_fiber: str = None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        main_type = FIBERS[main_fiber](self.gspace, planes, fixparams=True)
        inner_class = FIBERS[inner_fiber](self.gspace, planes, fixparams=True)
        if out_fiber is None:
            out_fiber = main_fiber
        out_type = FIBERS[out_fiber](self.gspace, planes, fixparams=True)
        
        for b, stride in enumerate(strides):
            if b == num_blocks - 1:
                out_f = out_type
            else:
                out_f = main_type
            layers.append(block(self.in_type, out_type, stride, out_fiber=out_f))
            self.in_type = layers[-1].out_type
            #print("Layer {}: {} -> {} -> {}".format(len(layers), in_type, main_type, out_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        if not self.exported:
            #x = pad(input, (0,1,0,1))
            x = enn.GeometricTensor(x, self.input_type)
        else:
            x = input
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.restrict2(out)
        out = self.layer2(out)
        out = self.restrict3(out)
        out = self.layer3(out)
        out = self.restrict4(out)
        out = self.layer4(out)
        if not self.exported:
            out = out.tensor
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def export(self):
        self.exported=True
        self.exported_model = self.model.export().eval()
    
    def unexport(self):
        self.exported=False

class SteerableBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, out_fiber = None):
        super(SteerableBasicBlock, self).__init__()
        if out_fiber is None:
            out_fiber = in_planes
        self.in_type = in_planes
        inner_class = planes
        self.out_type = out_fiber
        self.conv1 = enn.R2Conv(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = enn.InnerBatchNorm(planes)
        self.relu1 = enn.ReLU(planes,inplace=True)
        self.conv2 = enn.R2Conv(planes, out_fiber, 3, stride=1, padding=1, bias=False)
        self.bn2 = enn.InnerBatchNorm(out_fiber)
        self.relu2 = enn.ReLU(out_fiber, inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes.size != self.expansion * planes.size:
            self.shortcut = nn.Sequential(
                enn.R2Conv(in_planes, out_fiber, 1, stride=stride, bias=False),
                enn.InnerBatchNorm(out_fiber)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class SteerableBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, out_fiber = None):
        super(SteerableBottleneck, self).__init__()
        if out_fiber is None:
            out_fiber = in_planes
        self.conv1 = enn.R2Conv(in_planes, planes, 1, bias=False)
        self.bn1 = enn.InnerBatchNorm(planes)
        self.relu1 = enn.ReLU(planes,inplace=True)
        self.conv2 = enn.R2Conv(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = enn.R2Conv(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = enn.InnerBatchNorm(planes)
        self.relu2 = enn.ReLU(planes,inplace=True)
        self.conv3 = enn.R2Conv(planes, self.expansion * planes, 1, bias=False)
        self.relu3 = enn.ReLU(self.expansion * planes,inplace=True)
        self.bn3 = enn.InnerBatchNorm(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                enn.R2Conv(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                enn.InnerBatchNorm(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out

def SteerableResNet18():
    return SteerableResNet(SteerableBasicBlock, [2, 2, 2, 2])

def wrn28_10(num_classes=10):
    """Constructs a Wide ResNet 28-10 model.
    This model is only [R]otation equivariant (no flips equivariance)
    Args:
        pretrained (bool): If True, returns a model pre-trained on Cifar100
    """
    model = Wide_ResNet(depth=28, widen_factor=10, dropRate=0.3, num_classes=num_classes)
    return model