import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces
from e2cnn import nn as enn
from advbench.models.e2_utils import *
from torch.nn.functional import pad

class MNISTNet(nn.Module):
    def __init__(self, input_shape, num_classes, n_layers=2):
        super(MNISTNet, self).__init__()
        convs = [nn.Conv2d(input_shape[0], 32, 3, 1),nn.ReLU(),nn.Conv2d(32, 64, 3, 1),nn.ReLU()]
        for i in range(n_layers-2):
            convs.append(nn.Conv2d(64, 64, 3, 1)) 
            convs.append(nn.ReLU()) 
        self.convs = nn.Sequential(*convs)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        dim_out = input_shape[-1]-2*(n_layers)
        dim_out = ((int((dim_out-2)/2)+1)**2)*64
        self.fc1 = nn.Linear(dim_out, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.convs(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x    

class SteerableMNISTnet(torch.nn.Module):
    def __init__(self, n_classes=10, n_rot = 8, num_channels = 1, control_width = True):
        super(SteerableMNISTnet, self).__init__()
        self.exported=False
        channels = [32, 64]
        if control_width:
            channels = [int(c//(n_rot)) for c in channels]
        # the model is equivariant under rotations by 360/n_rot degrees, modelled by Cn_rot
        self.r2_act = gspaces.Rot2dOnR2(N=n_rot)
        
        # the input image is a scalars field, corresponding to the trivial representation
        in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr]*num_channels)
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 32 feature fields, each transforming under the regular representation of C8
        out_type_1 = enn.FieldType(self.r2_act, channels[0]*[self.r2_act.regular_repr])
        self.block1 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type_1, kernel_size=3, padding=1, bias=False),
            enn.ReLU(out_type_1, inplace=True)
        )
        in_type_2 = self.block1.out_type
        # the output type of the second convolution layer are 64 regular feature fields of C8
        out_type_2 = enn.FieldType(self.r2_act, channels[1]*[self.r2_act.regular_repr])
        self.block2 = enn.SequentialModule(
            enn.R2Conv(in_type_2, out_type_2, kernel_size=3, padding=1, bias=False),
            enn.ReLU(out_type_2, inplace=True)
        )
        self.pool1 = enn.PointwiseMaxPool(out_type_2, 2)
        out_type_size = self.pool1.out_type.size
        self.fc1 =  torch.nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(out_type_size*196, 128),
            nn.ReLU(inplace=True),
        )
        self.fc2 =  torch.nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)     
        )

        self.model = enn.SequentialModule(self.block1, self.block2, self.pool1)
    
    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        if not self.exported:
            #x = pad(input, (0,1,0,1))
            x = enn.GeometricTensor(input, self.input_type)
        else:
            x = input

        if not self.exported:
            x = self.model(x)
            # unwrap the output GeometricTensor
            # (take the Pytorch tensor and discard the associated representation)
            x = x.tensor
        else:
            x = self.exported_model(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def export(self):
        self.exported=True
        self.exported_model = self.model.export().eval()
    
    def unexport(self):
        self.exported=False


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