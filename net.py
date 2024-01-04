import mindspore as ms
from mindspore import nn,ops
import os
import mindcv


class BaseFeatureExtractor(nn.Cell):
    def construct(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def set_train(self, mode=True):
        # freeze BN mean and std
        for _,module in self.cells_and_names():
            if isinstance(module, ms.nn.BatchNorm2d):
                module.set_train(False)
            else:
                module.set_train(mode)


class ResNet50Fc(BaseFeatureExtractor):
    def __init__(self,model_path=None, normalize=True):
        super(ResNet50Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_resnet = mindcv.create_model("resnet50",pretrained=False)
                ms.load_checkpoint(model_path, self.model_resnet)
            else:
                raise Exception('invalid model path!')
        else:
            self.model_resnet = mindcv.create_model("resnet50",pretrained=False)

        if model_path or normalize:
            self.normalize = True
            self.register_buffer('mean', ms.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', ms.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features

    def construct(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class VGG16Fc(BaseFeatureExtractor):
    def __init__(self,model_path=None, normalize=True):
        super(VGG16Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_vgg = mindcv.create_model("vgg16",pretrained=False)
                ms.load_checkpoint(model_path, self.model_vgg)
            else:
                raise Exception('invalid model path!')
        else:
            self.model_vgg = mindcv.create_model("vgg16",pretrained=True)

        if model_path or normalize:
            self.normalize = True
            self.register_buffer('mean', ms.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', ms.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_vgg = self.model_vgg
        self.features = model_vgg.features
        self.classifier = nn.SequentialCell()
        for i in range(6):
            self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
        self.feature_layers = nn.SequentialCell(self.features, self.classifier)

        self.__in_features = 4096

    def construct(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features


class CLS(nn.Cell):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256, pretrain=False):
        super(CLS, self).__init__()
        self.pretrain = pretrain
        if bottle_neck_dim:
            self.bottleneck = nn.Dense(in_dim, bottle_neck_dim)
            self.fc = nn.Dense(bottle_neck_dim, out_dim)
            self.main = nn.SequentialCell(self.bottleneck,self.fc,nn.Softmax(axis=-1))
        else:
            self.fc = nn.Dense(in_dim, out_dim)
            self.main = nn.SequentialCell(self.fc,nn.Softmax(axis=-1))

    def construct(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out


class AdversarialNetwork(nn.Cell):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.SequentialCell(
            nn.Dense(in_feature, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Dense(1024,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Dense(1024, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def construct(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y
