from advbench.models.e2_networks import e2wrn
from advbench.models.e2_mnist import E2SFCNN, E2SFCNN_QUOT
from advbench.models.wrn import wrn16_8, wrn16_8_stl, wrn28_10
from advbench.models.resnet import ResNet18
from advbench.models.mnist import MNISTNet, CnSteerableCNN, SteerableMNISTnet
from advbench.models.dgcnn import DGCNN, cal_loss
import torch.nn as nn
from torch.nn.functional import cross_entropy
import torch.nn.init as init
import numpy as np
import timm

def Classifier(input_shape, num_classes, hparams, loss=None):
    model = create_model(input_shape, num_classes, hparams)
    if hparams["model"] == "DGCNN":
        loss = cal_loss
    elif 'label_smoothing' in hparams:
        loss = lambda pred, target, reduction='mean': cross_entropy(pred, target, reduction=reduction, label_smoothing=hparams['label_smoothing'])
    elif loss is None:
        # For all other models use CE
        loss = cross_entropy
    return ModelWrapper(model, loss)

def create_model(input_shape, num_classes, hparams):
    print("model", hparams["model"])
    if hparams["model"] == "DGCNN":
        model = DGCNN()
        return model
    elif input_shape[0] == 1:
        if hparams["model"] == "CnSteerableCNN":
            return CnSteerableCNN(num_channels=1)
        elif hparams["model"] == "SteerableCNN_C16_D16":
            return E2SFCNN(1, num_classes)
        elif hparams["model"] == "SteerableMNISTnet":
            return SteerableMNISTnet(n_classes = num_classes, num_channels = 1)
        elif hparams["model"] == "SteerableMNISTnet-exp":
            net =  SteerableMNISTnet(n_classes = num_classes, num_channels = 1)
            net.export()
            return net
        elif hparams["model"] == "CnSteerableCNN-exp":
            net = CnSteerableCNN(num_classes)
            net.export()
            return net
        elif hparams["model"] == "MNISTnet":
            if "n_layers" in hparams.keys():
                num_layers = int(hparams["n_layers"])
            else:
                num_layers = 2
            print(f"input shape {input_shape}, num classes {num_classes}, num layers {num_layers}")
            return MNISTNet(input_shape, num_classes, n_layers = num_layers)
        else:
            raise NotImplementedError
    elif input_shape[0] == 3:
        if hparams["model"] == "resnet18":
            return ResNet18(num_classes = num_classes)
        elif hparams["model"] == "wrn-28-7-rot":
            print("Using e2 invariant WRN-28-7")
            return e2wrn(depth=28, widen_factor = 7, num_classes=num_classes, r=3)
        elif hparams["model"] == "wrn-28-7-rot-d8":
            print("Using e2 invariant WRN-28-7")
            return e2wrn(depth=28, widen_factor = 7, num_classes=num_classes, r=-1)
        elif hparams["model"] == "wrn-28-10-rot":
            print("Using e2 invariant WRN-28-10")
            return e2wrn(depth=28, widen_factor = 10, num_classes=num_classes, r=3)
        elif hparams["model"] == "wrn-28-10":
            print("Using WRN-28-10")
            return wrn28_10(num_classes=num_classes)
        elif hparams["model"] == "wrn-16-8-stl":
            print("Using WRN-16-8")
            return wrn16_8_stl(num_classes=num_classes)
        elif hparams["model"] == "wrn-16-8":
            print("Using WRN-16-8")
            return wrn16_8(num_classes=num_classes)
        elif hparams["model"] == "wrn-16-8-rot":
            print("Using e2 invariant WRN-16-8-rot")
            return e2wrn(depth=16, widen_factor = 8, num_classes=num_classes, r=3, fixparams=False)
        elif hparams["model"] == "convnext-T":
            return timm.create_model('convnext_tiny', pretrained=True)
        elif hparams["model"] == "CnSteerableCNN":
            return CnSteerableCNN(num_channels=3, num_classes = num_classes)
        else:
            raise Exception("Unknown model: {}".format(hparams["model"]))
    else:
        raise Exception("Num channels not supported: {}".format(input_shape[0]))

class ModelWrapper(nn.Module):
    def __init__(self, model, loss):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, x):
        return self.model(x)

    def export(self):
        self.model.export()
    
    def unexport(self):
        self.model.unexport()
