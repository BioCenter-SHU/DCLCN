import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet50

from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        
        resnetModel = resnet50(pretrained=True)
        feature_map = list(resnetModel.children())
        feature_map.pop()
        self.feature_extractor = nn.Sequential(*feature_map)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        #self.classifier = nn.Sequential(
        #    nn.Linear(2048, 2),
        #)
        

    def forward(self, feature):
        feature = feature.expand(feature.data.shape[0], 3, 227, 227)#32,3,227,227
        feature = self.feature_extractor(feature)#32,2048,8,8
        feature = self.avgpool(feature)
        feature = feature.view(-1, 2048)
        #print(feature.shape)

        #reverse_bottleneck = grad_reverse.apply(feature, alpha)

        #class_output = self.classifier(feature)
        #domain_output = self.discriminator(reverse_bottleneck)

        return feature #, domain_output


'''class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x'''


class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        #self.fc1 = nn.Linear(8192, 3072)
        #self.bn1_fc = nn.BatchNorm1d(3072)
        #self.fc2 = nn.Linear(3072, 2048)
        #self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc1 = nn.Linear(2048, 2)
        self.bn_fc1 = nn.BatchNorm1d(2)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = gr.grad_reverse(x, self.lambd)
        #x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc1(x)
        x = self.bn_fc1(x)
       
        return x
