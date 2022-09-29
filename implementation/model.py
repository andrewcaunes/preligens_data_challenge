import torch.nn as nn
import torch
import torch.nn.functional as F

filters = 64
up_filters = 96
class Bn_cv(nn.Module):
    def __init__(self, input_channels=filters, output_channels=filters):
        super(Bn_cv, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.batchNorm = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.batchNorm(x)
        z = self.conv(z)
        z = self.relu(z)
        return z


class Bn_ucv(nn.Module):
    def __init__(self, input_channels=filters):
        super(Bn_ucv, self).__init__()
        self.convT = nn.ConvTranspose2d(input_channels, input_channels, kernel_size=3, stride=2, padding=(1,1), output_padding=1)
        self.batchNorm = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU()
        # self.mem = mem

    def forward(self, x):
        z = self.batchNorm(x)
        z = self.convT(z)
        z = self.relu(z)
        return z

class DownBlock(nn.Module):
    def __init__(self):
        super(DownBlock, self).__init__()
        self.bn_cv1 = Bn_cv(filters)
        self.bn_cv2 = Bn_cv(filters)
        self.bn_cv3 = Bn_cv(filters)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        ## block down
        z = self.bn_cv1(x)
        z = self.bn_cv2(z)
        saved_z = z
        z = self.bn_cv3(z)
        z = self.maxpool(z)
        return z, saved_z


class UpBlock(nn.Module):
    def __init__(self, input_channels=4):
        super(UpBlock, self).__init__()
        self.bn_cv1 = Bn_cv(filters*2, up_filters)
        self.bn_cv2 = Bn_cv(up_filters, filters)
        self.bn_ucv1 = Bn_ucv(filters)
    def forward(self, x, saved_z):
        ## block down
        # print("x",x.shape)
        # print("z",saved_z.shape)
        z = torch.cat((x, saved_z), dim=1)
        z = self.bn_cv1(z)
        z = self.bn_cv2(z)
        z = self.bn_ucv1(z)
        return z


class Block(nn.Module):
    def __init__(self, input_channels=4):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.down1 = DownBlock()
        self.down2 = DownBlock()
        self.down3 = DownBlock()
        self.down4 = DownBlock()
        self.up1 = UpBlock()
        self.up2 = UpBlock()
        self.up3 = UpBlock()
        self.up4 = UpBlock()
        self.bn_cv1 = Bn_cv(filters)
        self.bn_cv2 = Bn_cv(filters)
        self.bn_cv3 = Bn_cv(filters)
        self.bn_cv4 = Bn_cv(filters)
        self.bn_cv5 = Bn_cv(filters)
        self.bn_cv6 = Bn_cv(filters*2, up_filters)
        self.bn_cv7 = Bn_cv(up_filters, filters)
        self.bn_ucv1 = Bn_ucv(filters)
        self.conv2 = nn.Conv2d(filters, 10, kernel_size=1)
        self.batchNorm = nn.BatchNorm2d(input_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        # self.mem = mem

    def forward(self, x):
        z0 = self.conv1(x)
        z0 = self.relu(z0)
        z0 = self.bn_cv1(z0)
        z1 = self.bn_cv2(z0)
        z1 = self.maxpool(z1)

        ## block down
        z2, saved_z1 = self.down1(z1)
        z3, saved_z2 = self.down2(z2)
        z4, saved_z3 = self.down3(z3)
        z5, saved_z4 = self.down4(z4)
        z5 = self.bn_cv3(z5)
        z5 = self.bn_cv4(z5)
        z5 = self.bn_ucv1(z5)
        z4 = self.up1(z5, saved_z4)
        z3 = self.up2(z4, saved_z3)
        z2 = self.up3(z3, saved_z2)
        z1 = self.up4(z2, saved_z1)

        z0 = torch.cat((z1, z0), dim=1)
        z0 = self.bn_cv6(z0)
        z0 = self.bn_cv7(z0)
        z0 = self.conv2(z0)
        z0 = self.softmax(z0)
        ##



        return z0



############################

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = self.softmax(logits)
        return logits