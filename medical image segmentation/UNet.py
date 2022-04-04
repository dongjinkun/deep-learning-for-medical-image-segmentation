import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, drop_out=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        self.dropout = drop_out

    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x = nn.Dropout2d(0.5)(x)
        return x


class Up(nn.Module):  # UNet的上采样层
    def __init__(self, in_feat, out_feat, is_deconv=True):
        super(Up, self).__init__()

        if is_deconv:
            self.up = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
            self.conv = conv_block(in_feat, out_feat)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
            self.conv = conv_block(in_feat, out_feat)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, args, in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.args = args
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.out_size = args.out_size

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        # filters = [16, 32, 64, 128, 256]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # upsample
        self.up1 = Up(filters[4], filters[3], self.is_deconv)  # 256->128
        self.up2 = Up(filters[3], filters[2], self.is_deconv)  # 128->64
        self.up3 = Up(filters[2], filters[1], self.is_deconv)  # 64->32
        self.up4 = Up(filters[1], filters[0], self.is_deconv)  # 32->16
        # final conv (without any concat)
        self.final = nn.Sequential(nn.Conv2d(filters[0], n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):

        # Feature Extraction(Encoder)
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)

        # Upscaling Part (Decoder)
        x = self.up1(center, conv4)

        x = self.up2(x, conv3)

        x = self.up3(x, conv2)

        x = self.up4(x, conv1)

        out = self.final(x)

        return out

