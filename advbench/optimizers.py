import torch.optim as optim
import torch
from torch.nn.functional import relu
import torch.nn as nn
import e2cnn.nn as enn

#TODO(AR): Need to write an optimizer for primal-dual

def Optimizer(classifier, hparams):

    # return optim.Adadelta(
    #     classifier.parameters(),
    #     lr=1.0)

    return optim.SGD(
        classifier.parameters(),
        lr=hparams['learning_rate'],
        momentum=hparams['sgd_momentum'],
        weight_decay=hparams['weight_decay'])

class PrimalDualOptimizer:
    def __init__(self, parameters, margin, eta):
        self.parameters = parameters
        self.margin = margin
        self.eta = eta

    def step(self, cost):
        self.parameters['dual_var'] = relu(self.parameters['dual_var'] + self.eta * (cost - self.margin))

    @staticmethod
    def relu(x):
        return x if x > 0 else torch.tensor(0).cuda()

def SFCNN_Optimizer(model, hparams):
    # optimizer as in "Learning Steerable Filters for Rotation Equivariant CNNs"
    # https://arxiv.org/abs/1711.07289
    
    # split up parameters into groups, named_parameters() returns tuples ('name', parameter)
    # each group gets its own regularization gain
    batchnormLayers = [m for m in model.modules() if isinstance(m,
                                                                     (nn.modules.batchnorm.BatchNorm1d,
                                                                      nn.modules.batchnorm.BatchNorm2d,
                                                                      nn.modules.batchnorm.BatchNorm3d,
                                                                      enn.NormBatchNorm,
                                                                      enn.GNormBatchNorm,
                                                                      )
                                                                )]
    linearLayers = [m for m in model.modules() if isinstance(m, nn.modules.linear.Linear)]
    convlayers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, enn.R2Conv))]
    weights_conv = [p for m in convlayers for n, p in m.named_parameters() if n.endswith('weights') or n.endswith("weight")]
    biases = [p for n, p in model.named_parameters() if n.endswith('bias')]
    weights_bn = [p for m in batchnormLayers for n, p in m.named_parameters()
                  if n.endswith('weight') or n.split('.')[-1].startswith('weight')
                  ]
    weights_fully = [p for m in linearLayers for n, p in m.named_parameters() if n.endswith('weight')]
    # CROP OFF LAST WEIGHT !!!!! (classification layer)
    weights_fully, weights_softmax = weights_fully[:-1], [weights_fully[-1]]
    print("SFCNN optimizer")
    for n, p in model.named_parameters():
        if p.requires_grad and not n.endswith(('weight', 'weights', 'bias')):
            raise Exception('named parameter encountered which is neither a weight nor a bias but `{:s}`'.format(n))
    param_groups = [dict(params=weights_conv, lamb_L1=1e-7, lamb_L2=1e-7, weight_decay=1e-7),
                    dict(params=weights_bn, lamb_L1=0, lamb_L2=0, weight_decay=0),
                    dict(params=weights_fully, lamb_L1=1e-8, lamb_L2=1e-8, weight_decay=1e-8),
                    dict(params=weights_softmax, lamb_L1=0, lamb_L2=0, weight_decay=0),
                    dict(params=biases, lamb_L1=0, lamb_L2=0, weight_decay=0)]
    
    return optim.Adam(param_groups, lr=hparams['learning_rate'], betas=(0.9, 0.999))
