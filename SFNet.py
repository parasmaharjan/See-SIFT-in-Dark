import common
import torch.nn as nn
import torch

class SFNet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SFNet, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale
        act = nn.LeakyReLU(0.2, inplace=True)
        # act = nn.ReLU(True)
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.o_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x

class SFNetDownscaled(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SFNetDownscaled, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale
        act = nn.LeakyReLU(0.2, inplace=True)
        # act = nn.ReLU(True)
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.o_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.downscaled = nn.MaxPool2d(2, stride=2)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        x = self.downscaled(x)
        res = self.body(x)
        res += x

        x = self.tail(res)

        return x

class SFNetDownscaled4x(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SFNetDownscaled4x, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale
        act = nn.LeakyReLU(0.2, inplace=True)
        # act = nn.ReLU(True)
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        m_downscaled4x = [
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=2, bias=True, dilation=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=2, bias=True, dilation=2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=2, bias=True, dilation=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=2, bias=True, dilation=2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, stride=2)
        ]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.o_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.downscaled4x = nn.Sequential(*m_downscaled4x)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        x = self.downscaled4x(x)
        res = self.body(x)
        res += x

        x = self.tail(res)

        return x

    def lrelu(self, x):
        outt = torch.max(0.2*x, x)
        return outt

class MBSFNet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MBSFNet, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale
        act = nn.LeakyReLU(0.2, inplace=True)
        # act = nn.ReLU(True)
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail1 = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.o_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail1 = nn.Sequential(*m_tail1)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x1 = self.tail1(res)

        return x1

class MBSFNet18(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MBSFNet18, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale
        act = nn.LeakyReLU(0.2, inplace=True)
        # act = nn.ReLU(True)
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail1 = [
            common.Upsampler(conv, 4, n_feats, act=False),
            conv(n_feats, args.o_colors, kernel_size)
        ]
        m_tail2 = [
            common.Upsampler(conv, 2, n_feats, act=False),
            conv(n_feats, args.o_colors, kernel_size)
        ]
        m_tail3 = [
            #nn.MaxPool2d(2, stride=2),
            common.Upsampler(conv, 1, n_feats, act=False),
            conv(n_feats, args.o_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail1 = nn.Sequential(*m_tail1)
        self.tail2 = nn.Sequential(*m_tail2)
        self.tail3 = nn.Sequential(*m_tail3)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x1 = self.tail1(res)
        x2 = self.tail2(res)
        x3 = self.tail3(res)
        return x1, x2, x3