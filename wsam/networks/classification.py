import torch, torchvision
import torch.nn as nn
from torch.nn import init
from .Xception import *
from .InceptionV4 import *
from .WideResnet import *

def init_weights(m):
    global init_net
    inet = init_net.split(',')[0]
    dist = init_net.split(',')[1]
    if inet=='xavier':
        if dist=='uniform':
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None: m.bias.data.zero_()
        elif dist=='gauss':
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None: m.bias.data.zero_()
    if inet=='xavier':
        if dist=='uniform':
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight.data)
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None: m.bias.data.zero_()
        elif dist=='gauss':
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_ (m.weight.data)
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None: m.bias.data.zero_()

def ChooseNet(name, classes=10, pretrained=None):
    global init_net
    conf = pretrained
    init_net = conf.init # conf.init, pretrained is just for the env.
    if conf: 
        pretrained = False
    else: 
        pretrained = conf.pretrained;

    if name=="InceptionV3": # Only ImageNet input
        net = torchvision.models.inception_v3(num_classes=classes, pretrained=pretrained)
    elif name=="InceptionV4": # Only ImageNet input
        net = inceptionv4(num_classes=classes, pretrained=pretrained)
    elif name=="VGG16": # Only ImageNet input
        net = torchvision.models.vgg16_bn(pretrained=pretrained)
        net.classifier._modules['0'] = nn.Linear(8192, 4096) 
        net.classifier._modules['6'] = nn.Linear(4096, classes)
    elif name=="Resnet18":
        net = torchvision.models.resnet18(pretrained=pretrained)
        net.fc = nn.Linear(512,classes,bias=True)
    elif name=="Resnet50":
        net = torchvision.models.resnet50(pretrained=pretrained)
        net.fc = nn.Linear(2048,classes,bias=True)
    elif name=="Resnet101":
        net = torchvision.models.resnet101(pretrained=pretrained)
        net.fc = nn.Linear(2048,classes,bias=True)
    elif name=="Squeeze11":
        net = torchvision.models.squeezenet1_1(pretrained=pretrained)
        net.num_classes=classes
        net.classifier._modules['1'] = nn.Conv2d(512, classes, kernel_size=(1, 1), stride=(1, 1))
        net.classifier._modules['3'] = nn.AvgPool2d(kernel_size=5, stride=1, padding=0)
    elif name=="WideResNet101":
        # net = torchvision.models.wide_resnet101_2(num_classes=classes, pretrained=pretrained)
        dropFC=0.3
        net = WideResNet(28, 10, dropout_rate=dropFC, num_classes=classes)
    elif name=="Xception":
        net = xception(num_classes=classes, pretrained=pretrained)
    if not pretrained:
        print(name, " , ", init_net )
        net.apply(init_weights)
    return net
