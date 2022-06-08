import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary


class FC_layer(nn.Module):
    def __init__(self, class_num=10):
        super(FC_layer, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, class_num)
        )

    def forward(self, x):
        return self.fc(x)


class Plain_conv(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Plain_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        return F.relu(self.bn1(self.conv1(x)), inplace=True)


class Plain_change_channel(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Plain_change_channel, self).__init__()
        self.change_channel = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        return self.change_channel(x)


class Plain34(nn.Module):
    def __init__(self, class_num=10):
        super(Plain34, self).__init__()
        self.prepare = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.conv64 = nn.Sequential(
            Plain_conv(64, 64, 1),
            Plain_conv(64, 64, 1),
            Plain_conv(64, 64, 1),
            Plain_conv(64, 64, 1),
            Plain_conv(64, 64, 1),
            Plain_conv(64, 64, 1),
        )
        self.conv128 = nn.Sequential(
            Plain_change_channel(64, 128, 2),
            Plain_conv(128, 128, 1),
            Plain_conv(128, 128, 1),
            Plain_conv(128, 128, 1),
            Plain_conv(128, 128, 1),
            Plain_conv(128, 128, 1),
            Plain_conv(128, 128, 1),
            Plain_conv(128, 128, 1),
        )
        self.conv256 = nn.Sequential(
            Plain_change_channel(128, 256, 2),
            Plain_conv(256, 256, 1),
            Plain_conv(256, 256, 1),
            Plain_conv(256, 256, 1),
            Plain_conv(256, 256, 1),
            Plain_conv(256, 256, 1),
            Plain_conv(256, 256, 1),
            Plain_conv(256, 256, 1),
            Plain_conv(256, 256, 1),
            Plain_conv(256, 256, 1),
            Plain_conv(256, 256, 1),
            Plain_conv(256, 256, 1),
        )
        self.conv512 = nn.Sequential(
            Plain_change_channel(256, 512, 2),
            Plain_conv(512, 512, 1),
            Plain_conv(512, 512, 1),
            Plain_conv(512, 512, 1),
            Plain_conv(512, 512, 1),
            Plain_conv(512, 512, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = FC_layer(class_num=class_num)

    def forward(self, x):
        x = self.prepare(x)
        x = self.conv64(x)
        x = self.conv128(x)
        x = self.conv256(x)
        x = self.conv512(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x += identity
        return F.relu(x, inplace=True)


class SpecialBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(SpecialBlock, self).__init__()
        self.change_channel = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = self.change_channel(x)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x += identity
        return F.relu(x, inplace=True)


class ResNet34(nn.Module):
    def __init__(self, class_num=10):
        super(ResNet34, self).__init__()
        self.prepare = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = nn.Sequential(
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            SpecialBlock(128, 256, [2, 1]),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1)
        )
        self.layer4 = nn.Sequential(
            SpecialBlock(256, 512, [2, 1]),
            CommonBlock(512, 512, 1),
            CommonBlock(512, 512, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = FC_layer(class_num=class_num)

    def forward(self, x):
        x = self.prepare(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.pool3(x)
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.pool4(x)
        x = self.relu(self.conv13(x))
        x = self.relu(self.conv14(x))
        x = self.relu(self.conv15(x))
        x = self.relu(self.conv16(x))
        x = self.pool5(x)
        # x = x.view(x.size()[0], -1)
        x = x.view(-1, 7 * 7 * 512)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        output = self.softmax(self.fc3(x))
        return output


if __name__ == '__main__':
    vgg19_net = VGG19()
    resnet34_net = ResNet34(class_num=10)
    plain34_net = Plain34(class_num=10)
    data = torch.ones(size=(10, 3, 224, 224))
    print('___________VGG19____________')
    vgg19_net(data)
    print(vgg19_net(data).shape)
    print(vgg19_net(data))
    # summary(net)
    print('___________________________')
    print('___________ResNet34____________')
    print(resnet34_net)
    resnet34_net(data)
    print(resnet34_net(data).shape)
    print(resnet34_net(data))
    summary(resnet34_net)
    print('___________________________')
    print('___________Plain34____________')
    print(plain34_net)
    plain34_net(data)
    print(plain34_net(data).shape)
    print(plain34_net(data))
    summary(plain34_net)
    print('___________________________')