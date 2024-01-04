from mindspore.common.initializer import initializer, HeNormal , XavierNormal,Normal
import mindspore as ms
from mindspore import nn,ops

import numpy as np
import os


def forw(ctx, coeff, input):
    ctx.coeff = coeff
    return input.view_as(input)

def backw(ctx, grad_outputs):
    coeff = ctx.coeff
    return None, -coeff * grad_outputs

def bprop():
    op = ops.Custom(backw, lambda x, _: x, lambda x, _: x, func_type="akg")
    def custom_bprop(x, out, dout):
        dx = op(x, dout)
        return (dx,)

    return custom_bprop

class MyGradientReverseLayer(nn.Cell):
    def __int__(self):
        super().__int__()
        self.op = ops.Custom(forw, lambda x: x, lambda x: x, bprop=bprop(), func_type="akg")
    def construct(self, ctx, coeff, input):
        return self.op(ctx, coeff, input)



class MyGradientReverseModule(nn.Cell):
    def __init__(self, scheduler):
        super(MyGradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.register_buffer('global_step', ops.zeros(1))
        self.coeff = 0.0
        self.grl = MyGradientReverseLayer.apply

    def construct(self, x):
        if self.global_step.item() < 25000:
            self.coeff = 0
        else:
            self.coeff = self.scheduler(self.global_step.item() - 25000)
        if self.training:
            self.global_step += 1.0
        return self.grl(self.coeff, x)

def conv3x3(i_c, o_c, stride=1):
    return nn.Conv2d(i_c, o_c, 3, stride,"valid", 1, has_bias=False)

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, channels, momentum=1e-3, eps=1e-3):
        super().__init__(channels)
        self.update_batch_stats = True

    def construct(self, x):
        if self.update_batch_stats:
            return super().construct(x)
        else:
            return ops.batch_norm(
                x, None, None, self.weight, self.bias, True, self.momentum, self.eps
            )

def relu():
    return nn.LeakyReLU(0.1)

class residual(nn.Cell):
    def __init__(self, input_channels, output_channels, stride=1, activate_before_residual=False):
        super().__init__()
        layer = []
        if activate_before_residual:
            self.pre_act = nn.SequentialCell(
                BatchNorm2d(input_channels),
                relu()
            )
        else:
            self.pre_act = nn.Identity()
            layer.append(BatchNorm2d(input_channels))
            layer.append(relu())
        layer.append(conv3x3(input_channels, output_channels, stride))
        layer.append(BatchNorm2d(output_channels))
        layer.append(relu())
        layer.append(conv3x3(output_channels, output_channels))

        if stride >= 2 or input_channels != output_channels:
            self.identity = nn.Conv2d(input_channels, output_channels, 1, stride, has_bias=False)
        else:
            self.identity = nn.Identity()

        self.layer = nn.SequentialCell(*layer)

    def construct(self, x):
        x = self.pre_act(x)
        return self.identity(x) + self.layer(x)

class testSvhnTransfer(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(in_channels=3,out_channels=1,kernel_size=1,stride=1,padding=0)
    def construct(self,x):
        x=self.conv(x)
        return x

class WRN(nn.Cell):
    """ WRN28-width with leaky relu (negative slope is 0.1)"""
    def __init__(self, width, num_classes, transform_fn=None):
        super().__init__()

        self.init_conv = conv3x3(3, 16)

        filters = [16, 16*width, 32*width, 64*width]

        unit1 = [residual(filters[0], filters[1], activate_before_residual=True)] + \
            [residual(filters[1], filters[1]) for _ in range(1, 4)]
        self.unit1 = nn.SequentialCell(*unit1)

        unit2 = [residual(filters[1], filters[2], 2)] + \
            [residual(filters[2], filters[2]) for _ in range(1, 4)]
        self.unit2 = nn.SequentialCell(*unit2)

        unit3 = [residual(filters[2], filters[3], 2)] + \
            [residual(filters[3], filters[3]) for _ in range(1, 4)]
        self.unit3 = nn.SequentialCell(*unit3)

        self.unit4 = nn.SequentialCell(*[BatchNorm2d(filters[3]), relu(), nn.AdaptiveAvgPool2d(1)])

        self.cls = nn.SequentialCell(
            nn.Dense(filters[3], num_classes)
        )

        for _,m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(initializer(HeNormal(),m.weight.shape,m.weight.dtype))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(initializer("ones",m.gamma.shape,m.gamma.dtype))
                m.beta.set_data(initializer("ones", m.beta.shape, m.beta.dtype))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(initializer(XavierNormal()),m.weight.shape,m.weight.dtype)
                m.bias.set_data(initializer("zeros",m.bias.shape,m.bias.dtype))


        self.transform_fn = transform_fn
        self.__in_features = filters[3]

    def construct(self, x, return_feature=True):
        if self.training and self.transform_fn is not None:
            x = self.transform_fn(x)
        x = self.init_conv(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.unit3(x)
        f = self.unit4(x)
        f = ops.tanh(f)
        f=f.squeeze()
        c = self.cls(f)

        return f, c

    def update_batch_stats(self, flag):
        for _,m in self.cells_and_names():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag


class WRN2(nn.Cell):
    """ WRN28-width with leaky relu (negative slope is 0.1)"""

    def __init__(self, width, num_classes, transform_fn=None):
        super().__init__()

        self.init_conv = conv3x3(3, 16)

        filters = [16, 16 * width, 32 * width, 64 * width]

        unit1 = [residual(filters[0], filters[1], activate_before_residual=True)] + \
                [residual(filters[1], filters[1]) for _ in range(1, 4)]
        self.unit1 = nn.SequentialCell(*unit1)

        unit2 = [residual(filters[1], filters[2], 2)] + \
                [residual(filters[2], filters[2]) for _ in range(1, 4)]
        self.unit2 = nn.SequentialCell(*unit2)

        unit3 = [residual(filters[2], filters[3], 2)] + \
                [residual(filters[3], filters[3]) for _ in range(1, 4)]
        self.unit3 = nn.SequentialCell(*unit3)

        self.unit4 = nn.SequentialCell(*[BatchNorm2d(filters[3]), relu(), nn.AdaptiveAvgPool2d(1)])

        self.cls = nn.SequentialCell(
            nn.Dense(filters[3], num_classes)
        )


        for _,m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(initializer(HeNormal(),m.weight.shape,m.weight.dtype))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(initializer("ones",m.gamma.shape,m.gamma.dtype))
                m.beta.set_data(initializer("ones", m.beta.shape, m.beta.dtype))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(initializer(XavierNormal()),m.weight.shape,m.weight.dtype)
                m.bias.set_data(initializer("zeros",m.bias.shape,m.bias.dtype))

        self.transform_fn = transform_fn
        self.__in_features = filters[3]

    def construct(self, x, return_feature=True):
        if self.training and self.transform_fn is not None:
            x = self.transform_fn(x)
        x = self.init_conv(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.unit3(x)
        f = self.unit4(x)
        f = ops.tanh(f)
        f = f.squeeze()
        c = self.cls(f)

        return f

    def update_batch_stats(self, flag):
        for _,m in self.cells_and_names():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag

class WRN_C(nn.Cell):
    """ WRN28-width with leaky relu (negative slope is 0.1)"""
    def __init__(self, width, num_classes, transform_fn=None):
        super().__init__()

        self.init_conv = conv3x3(3, 16)

        filters = [16, 16*width, 32*width, 64*width]

        unit1 = [residual(filters[0], filters[1], activate_before_residual=True)] + \
            [residual(filters[1], filters[1]) for _ in range(1, 4)]
        self.unit1 = nn.SequentialCell(*unit1)

        unit2 = [residual(filters[1], filters[2], 2)] + \
            [residual(filters[2], filters[2]) for _ in range(1, 4)]
        self.unit2 = nn.SequentialCell(*unit2)

        unit3 = [residual(filters[2], filters[3], 2)] + \
            [residual(filters[3], filters[3]) for _ in range(1, 4)]
        self.unit3 = nn.SequentialCell(*unit3)

        self.unit4 = nn.SequentialCell(*[BatchNorm2d(filters[3]), relu(), nn.AdaptiveAvgPool2d(1)])

        self.output = nn.Dense(filters[3], num_classes)

        for _,m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(initializer(HeNormal(),m.weight.shape,m.weight.dtype))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(initializer("ones",m.gamma.shape,m.gamma.dtype))
                m.beta.set_data(initializer("ones", m.beta.shape, m.beta.dtype))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(initializer(XavierNormal()),m.weight.shape,m.weight.dtype)
                m.bias.set_data(initializer("zeros",m.bias.shape,m.bias.dtype))

        self.transform_fn = transform_fn

    def construct(self, x, return_feature=False):
        if self.training and self.transform_fn is not None:
            x = self.transform_fn(x)
        x = self.init_conv(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.unit3(x)
        f = self.unit4(x)
        c = self.output(f.squeeze())
        if return_feature:
            return [c, f]
        else:
            return c

    def update_batch_stats(self, flag):
        for _,m in self.cells_and_names():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag

class WRN_C2(nn.Cell):
    """ WRN28-width with leaky relu (negative slope is 0.1)"""
    def __init__(self, width, num_classes, transform_fn=None):
        super().__init__()

        self.init_conv = conv3x3(2, 16)

        filters = [16, 16*width, 32*width, 64*width]

        unit1 = [residual(filters[0], filters[1], activate_before_residual=True)] + \
            [residual(filters[1], filters[1]) for _ in range(1, 4)]
        self.unit1 = nn.SequentialCell(*unit1)

        unit2 = [residual(filters[1], filters[2], 2)] + \
            [residual(filters[2], filters[2]) for _ in range(1, 4)]
        self.unit2 = nn.SequentialCell(*unit2)

        unit3 = [residual(filters[2], filters[3], 2)] + \
            [residual(filters[3], filters[3]) for _ in range(1, 4)]
        self.unit3 = nn.SequentialCell(*unit3)

        self.unit4 = nn.SequentialCell(*[BatchNorm2d(filters[3]), relu(), nn.AdaptiveAvgPool2d(1)])

        self.output = nn.Dense(filters[3], num_classes)

        for _,m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(initializer(HeNormal(),m.weight.shape,m.weight.dtype))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(initializer("ones",m.gamma.shape,m.gamma.dtype))
                m.beta.set_data(initializer("ones", m.beta.shape, m.beta.dtype))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(initializer(XavierNormal()),m.weight.shape,m.weight.dtype)
                m.bias.set_data(initializer("zeros",m.bias.shape,m.bias.dtype))

        self.transform_fn = transform_fn

    def construct(self, x, return_feature=False):
        if self.training and self.transform_fn is not None:
            x = self.transform_fn(x)
        x = self.init_conv(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.unit3(x)
        f = self.unit4(x)
        c = self.output(f.squeeze())
        if return_feature:
            return [c, f]
        else:
            return c

    def update_batch_stats(self, flag):
        for _,m in self.cells_and_names():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag




class adversarialnet(nn.Cell):
    """
    Discriminator network.
    """
    def __init__(self, in_feature):
        super(adversarialnet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=20000))

    def construct(self, x):
        x = self.grl(x)
        y = self.main(x)
        y=ops.clamp(y,min=1e-2,max=1-1e-2)
        return y

class adversarialnet_c(nn.Cell):
    """
    Discriminator network.
    """
    def __init__(self, in_feature):
        super(adversarialnet_c, self).__init__()
        self.bottleneck = nn.SequentialCell(
            nn.Dense(in_feature, 128))
        self.main = nn.SequentialCell(
            nn.Dense(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Dense(256,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Dense(256, 256),
            nn.Dense(256, 6)
        )
        self.sig=nn.Sigmoid()
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=20000))

    def construct(self, x):
        x1=self.bottleneck(x)
        x2 = self.main(x1)
        x3=self.sig(x2)
        return x,x1,x2,x3

class AdversarialNetwork(nn.Cell):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Dense(in_feature, hidden_size)
        self.ad_layer2 = nn.Dense(hidden_size, hidden_size)
        self.ad_layer3 = nn.Dense(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def construct(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters_dict(), "lr_mult": 10, 'decay_mult': 2}]

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        m.weight.set_data(initializer(HeNormal(), m.weight.shape, m.weight.dtype))
        m.bias.set_data(initializer("zeros", m.bias.shape, m.bias.dtype))
    elif classname.find('BatchNorm') != -1:
        m.gamma.set_data(initializer("ones", m.gamma.shape, m.gamma.dtype))
        m.beta.set_data(initializer("ones", m.beta.shape, m.beta.dtype))
    elif classname.find('Linear') != -1:
        m.weight.set_data(initializer(XavierNormal()), m.weight.shape, m.weight.dtype)
        m.bias.set_data(initializer("zeros", m.bias.shape, m.bias.dtype))




class BaseFeatureExtractor(nn.Cell):
    def construct(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def set_train(self, mode=True):
        for _,module in self.cells_and_names():
            if isinstance(module, nn.BatchNorm2d):
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
        for _,module in self.main.cells_and_names():
            x = module(x)
            out.append(x)
        return out

class CLS2(nn.Cell):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256, pretrain=False):#128
        super(CLS2, self).__init__()
        self.pretrain = pretrain
        if bottle_neck_dim:
            self.bottleneck = nn.SequentialCell(nn.Dense(in_dim, bottle_neck_dim)

                                            )
            self.fc = nn.SequentialCell(nn.Linear(bottle_neck_dim,bottle_neck_dim),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Dense(bottle_neck_dim, bottle_neck_dim),#
                                    nn.ReLU(),#
                                    nn.Dropout(0.5),
                                    nn.Dense(bottle_neck_dim, out_dim)
                                    )
            self.main = nn.SequentialCell(self.bottleneck,self.fc,nn.Softmax(axis=-1))
        else:
            self.fc = nn.Dense(in_dim, out_dim)
            self.main = nn.SequentialCell(self.fc,nn.Softmax(axis=-1))


    def construct(self, x):
        out = [x]
        for _,module in self.main.cells_and_names():
            x = module(x)
            out.append(x)
        return out

class ResidualBlock(nn.Cell):
    def __init__(self, in_features: int):

        super().__init__()


        self.block = nn.SequentialCell(
        nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, padding_mode='reflect'),
        nn.InstanceNorm2d(in_features),
        nn.ReLU(),
        nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, padding_mode='reflect'),
        nn.InstanceNorm2d(in_features),
        nn.ReLU(),)

    def construct(self, x: ms.Tensor):
        return x + self.block(x)

class GeneratorResNetxy(nn.Cell):
#
    def __init__(self, input_channels: int, n_residual_blocks: int):
        super().__init__()

        out_features = 1

        layers = [
                        nn.Conv2d(input_channels, out_features, kernel_size=7, padding=3, padding_mode='reflect'),#kernel_size=7,padding=3
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(),
            ]

        in_features = out_features
        for _ in range(2):

            out_features *= 2

            layers += [
                            nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                            nn.InstanceNorm2d(out_features),
                            nn.ReLU(),
                        ]

            in_features = out_features

        for _ in range(n_residual_blocks):

            layers += [ResidualBlock(out_features)]

        self.layers = nn.SequentialCell(*layers)
        self.apply(weights_init_normal)


    def construct(self, x):

        x = self.layers(x)
        x = x.view(-1, 4 * 8 * 8)
        return x

class GeneratorResNetxy_MNIST(nn.Cell):
#
    def __init__(self, input_channels: int, n_residual_blocks: int):
        super().__init__()

        out_features = 1#64

        layers = [
                        nn.Conv2d(input_channels, out_features, kernel_size=7, padding=3, padding_mode='reflect'),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(),
            ]

        in_features = out_features
#then, down-sampled
        for _ in range(2):

            out_features *= 2

            layers += [
                            nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                            nn.InstanceNorm2d(out_features),
                            nn.ReLU(),
                        ]

            in_features = out_features

        for _ in range(n_residual_blocks):

            layers += [ResidualBlock(out_features)]


        self.layers = nn.SequentialCell(*layers)
        self.apply(weights_init_normal)


    def construct(self, x):

        x = self.layers(x)
        x = x.view(-1, 4 * 7 * 7)
        return x


class GeneratorResNetyx(nn.Cell):
#
    def __init__(self, input_channels: int):
        super().__init__()

        out_features = 64*4
        in_features=2

        layers = []


        for _ in range(2):

            out_features //= 2

            layers += [
                            nn.Upsample(scale_factor=2),#2
                            nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
                            nn.InstanceNorm2d(out_features),
                            nn.ReLU(),
                   ]

            in_features = out_features

        layers += [nn.Conv2d(out_features, input_channels, 7, padding=3, padding_mode='reflect'), nn.Tanh()]

        self.layers = nn.SequentialCell(*layers)
        self.apply(weights_init_normal)


    def construct(self, x):
        x = x.view(-1,2, 8 , 8)
        x=self.layers(x)

        return x

class GeneratorResNetyx_MNIST(nn.Cell):
#
    def __init__(self, input_channels: int):
        super().__init__()

        out_features = 64*4
        in_features=4

        layers = []

        for _ in range(2):

            out_features //= 2

            layers += [
                            nn.Upsample(scale_factor=2),#2
                            nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
                            nn.InstanceNorm2d(out_features),
                            nn.ReLU(),
                   ]

            in_features = out_features

        layers += [nn.Conv2d(out_features, input_channels, 7, padding=3, padding_mode='reflect'), nn.Tanh()]

        self.layers = nn.SequentialCell(*layers)
        self.apply(weights_init_normal)


    def construct(self, x):
        x = x.view(-1,4,  7 , 7)
        x=self.layers(x)

        return x


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.set_data(initializer(Normal(sigma=0.02,mean=0.0),shape=m.weight.shape,dtype=m.weight.dtype))