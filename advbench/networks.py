import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
from e2cnn import gspaces
from e2cnn import nn as enn
from einops import rearrange
from torch.nn.functional import pad

def Classifier(input_shape, num_classes, hparams):
    if input_shape[0] == 1:
        # return SmallCNN()
        return CnSteerableCNN(num_classes)#MNISTNet(input_shape, num_classes)
    elif input_shape[0] == 3:
        # return models.resnet18(num_classes=num_classes)
        return ResNet18()
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


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


class CnSteerableCNN(torch.nn.Module):
    
    def __init__(self, n_classes=10, n_rot = 8):
        
        super(CnSteerableCNN, self).__init__()
        self.exported=False
        
        # the model is equivariant under rotations by 360/n_rot degrees, modelled by Cn_rot
        self.r2_act = gspaces.Rot2dOnR2(N=n_rot)
        
        # the input image is a scalars field, corresponding to the trivial representation
        in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
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