from models.mfmsNet import *


class _DenseAsppBlock(nn.Module):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out):
        super(_DenseAsppBlock, self).__init__()
        self.tmp_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
            nn.BatchNorm2d(num1, momentum=0.0003),
            nn.ReLU(inplace=True)
        )
        self.dilate_layer = nn.Sequential(
            nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3, dilation=dilation_rate, padding=dilation_rate),
            nn.BatchNorm2d(num2, momentum=0.0003),
            nn.ReLU(inplace=True)
        )

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = self.tmp_layer(_input)
        feature = self.dilate_layer(feature)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature


class _DenseAsppBlockV2(nn.Module):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, k, dilation_rate, drop_out):
        super(_DenseAsppBlockV2, self).__init__()
        self.tmp_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
            nn.BatchNorm2d(num1, momentum=0.0003),
            nn.ReLU(inplace=True)
        )
        pad = (k + (k-1) * (dilation_rate-1) - 1) // 2
        self.dilate_layer = nn.Sequential(
            nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=k, dilation=dilation_rate, padding=pad),
            nn.BatchNorm2d(num2, momentum=0.0003),
            nn.ReLU(inplace=True)
        )

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = self.tmp_layer(_input)
        feature = self.dilate_layer(feature)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature


class _denseASPP(nn.Module):
    def __init__(self, channel, dilates=(3, 6, 12, 18, 24), reduce_channel_times=2, drop_out=0):
        super(_denseASPP, self).__init__()
        aspp_layers = []
        num_channels = max(channel//reduce_channel_times, 32)
        d_feature1 = 64  # 128
        d_feature2 = 64
        self.init_layer = conv1x1_bn_relu(channel, num_channels)
        for i, d in enumerate(dilates):
            aspp_layers.append(_DenseAsppBlock(
                input_num=num_channels+i*d_feature2,
                num1=d_feature1,
                num2=d_feature2,
                dilation_rate=d,
                drop_out=drop_out))
        self.denselayers = nn.ModuleList(aspp_layers)
        self.weight_layer = nn.Sequential(
            nn.Conv2d(num_channels + len(dilates) * d_feature2, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.init_layer(x)
        for layer in self.denselayers:
            feat = layer(features)
            features = torch.cat((features, feat), dim=1)

        w = self.weight_layer(features)
        x = w * x + x
        return x


class denseASPP(nn.Module):
    def __init__(self, channel, dilates=(3, 6, 12, 18, 24), reduce_channel_times=2, drop_out=0):
        super(denseASPP, self).__init__()
        aspp_layers = []
        num_channels = max(channel//reduce_channel_times, 32)
        d_feature1 = 64  # 128
        d_feature2 = 64
        self.init_layer = conv1x1_bn_relu(channel, num_channels)
        for i, d in enumerate(dilates):
            aspp_layers.append(_DenseAsppBlock(
                input_num=num_channels+i*d_feature2,
                num1=d_feature1,
                num2=d_feature2,
                dilation_rate=d,
                drop_out=drop_out))
        self.denselayers = nn.ModuleList(aspp_layers)
        # self.weight_layer = nn.Sequential(
        #     nn.Conv2d(num_channels+len(dilates)*d_feature2, channel, kernel_size=1),
        #     nn.BatchNorm2d(channel),
        #     nn.ReLU(inplace=True))

        self.last_layer = nn.Sequential(
            nn.Conv2d(num_channels + len(dilates) * d_feature2, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        features = self.init_layer(x)
        for layer in self.denselayers:
            feat = layer(features)
            features = torch.cat((features, feat), dim=1)

        # w = self.weight_layer(feat)
        # x = w * x + x
        x = self.last_layer(features)
        return x


class denseASPPV2(nn.Module):
    def __init__(self, channel, k, dilates=(3, 6, 12, 18, 24), reduce_channel_times=2, drop_out=0):
        super(denseASPPV2, self).__init__()
        aspp_layers = []
        num_channels = max(channel//reduce_channel_times, 32)
        d_feature1 = 64  # 128
        d_feature2 = 64
        self.init_layer = conv1x1_bn_relu(channel, num_channels)
        for i, d in enumerate(dilates):
            aspp_layers.append(_DenseAsppBlockV2(
                input_num=num_channels+i*d_feature2,
                num1=d_feature1,
                num2=d_feature2,
                k=k,
                dilation_rate=d,
                drop_out=drop_out))
        self.denselayers = nn.ModuleList(aspp_layers)
        # self.weight_layer = nn.Sequential(
        #     nn.Conv2d(d_feature2, channel, kernel_size=1),
        #     nn.BatchNorm2d(channel),
        #     nn.Sigmoid()
        # )
        self.last_layer = nn.Sequential(
            nn.Conv2d(num_channels + len(dilates) * d_feature2, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        features = self.init_layer(x)
        for layer in self.denselayers:
            feat = layer(features)
            features = torch.cat((features, feat), dim=1)

        # w = self.weight_layer(feat)
        # x = w * x + x
        x = self.last_layer(features)
        return x


class MultiFieAtConv2d_M2(nn.Module):
    def __init__(self, channels, dilates):
        super(MultiFieAtConv2d_M2, self).__init__()
        # self.short_field_conv = ShortFieldConv2d(in_channels=channels, out_channels=channels)
        self.mid_field_conv = _denseASPP(channels, dilates)
        # self.global_field_conv = GlobalFieldConv2d(in_channels=channels, channels=channels)
        self.merge_conv = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        # s = self.short_field_conv(x)
        m = self.mid_field_conv(x)
        # g = self.global_field_conv(x)
        merge = self.merge_conv(torch.cat((x, m), dim=1))
        return merge


class MultiFieAtConv2d_M3(nn.Module):
    def __init__(self, channels, k, dilates):
        super(MultiFieAtConv2d_M3, self).__init__()
        # self.short_field_conv = ShortFieldConv2d(in_channels=channels, out_channels=channels)
        self.mid_field_conv = denseASPPV2(channels, k, dilates)
        # self.global_field_conv = GlobalFieldConv2d(in_channels=channels, channels=channels)
        self.merge_conv = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        # s = self.short_field_conv(x)
        m = self.mid_field_conv(x)
        # g = self.global_field_conv(x)
        merge = self.merge_conv(torch.cat((x, m), dim=1))
        return merge


class MultiFieAtConv2d_SMLC(nn.Module):
    def __init__(self, channels, dilates):
        super(MultiFieAtConv2d_SMLC, self).__init__()
        self.s_field_conv = ShortFieldConv2d(in_channels=channels, out_channels=channels)
        self.m_field_conv = MidLongFieldConv2d(channels=channels)
        self.l_field_conv = denseASPP(channels, dilates)
        self.c_field_conv = GlobalFieldConv2d(in_channels=channels, channels=channels)
        self.merge_conv = nn.Sequential(
            nn.Conv2d(channels*5, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        s = self.s_field_conv(x)
        m = self.m_field_conv(x)
        l = self.l_field_conv(x)
        c = self.c_field_conv(x)
        merge = self.merge_conv(torch.cat((x, s, m, l, c), dim=1))
        return merge


class MultiFieAtConv2d_SM(nn.Module):
    def __init__(self, channels):
        super(MultiFieAtConv2d_SM, self).__init__()
        self.s_field_conv = ShortFieldConv2d(in_channels=channels, out_channels=channels)
        self.m_field_conv = MidLongFieldConv2d(channels=channels)
        # self.l_field_conv = denseASPP(channels, dilates)
        # self.c_field_conv = GlobalFieldConv2d(in_channels=channels, channels=channels)
        self.merge_conv = nn.Sequential(
            nn.Conv2d(channels*3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        s = self.s_field_conv(x)
        m = self.m_field_conv(x)
        # l = self.l_field_conv(x)
        # c = self.c_field_conv(x)
        merge = self.merge_conv(torch.cat((x, s, m), dim=1))
        return merge


class MultiFieAtConv2d_SM2(nn.Module):
    def __init__(self, channels):
        super(MultiFieAtConv2d_SM2, self).__init__()
        self.s_field_conv = ShortFieldConv2d(in_channels=channels, out_channels=channels)
        self.m_field_conv = MidLongFieldConv2d(channels=channels)
        # self.l_field_conv = denseASPP(channels, dilates)
        # self.c_field_conv = GlobalFieldConv2d(in_channels=channels, channels=channels)
        self.merge_conv = nn.Sequential(
            nn.Conv2d(channels*3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        s = self.s_field_conv(x)
        m = self.m_field_conv(s)
        # l = self.l_field_conv(x)
        # c = self.c_field_conv(x)
        merge = self.merge_conv(torch.cat((x, s, m), dim=1))
        return merge


class MultiFieAtConv2d_MC(nn.Module):
    def __init__(self, channels):
        super(MultiFieAtConv2d_MC, self).__init__()
        # self.s_field_conv = ShortFieldConv2d(in_channels=channels, out_channels=channels)
        self.m_field_conv = MidLongFieldConv2d(channels=channels)
        # self.l_field_conv = denseASPP(channels, dilates)
        self.c_field_conv = GlobalFieldConv2d(in_channels=channels, channels=channels)
        self.merge_conv = nn.Sequential(
            nn.Conv2d(channels*3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        # s = self.s_field_conv(x)
        m = self.m_field_conv(x)
        # l = self.l_field_conv(x)
        c = self.c_field_conv(x)
        merge = self.merge_conv(torch.cat((x, m, c), dim=1))
        return merge


class MultiFieAtConv2d_Combin(nn.Module):
    def __init__(self, channels):
        super(MultiFieAtConv2d_Combin, self).__init__()
        self.short_field_conv = ShortFieldConv2d(in_channels=channels, out_channels=channels)
        self.mid_field_conv = MidLongFieldConv2d(channels=channels)
        self.global_field_conv = SAM(in_dim=channels)
        self.chan_field_conv = GlobalFieldConv2d(in_channels=channels, channels=channels)
        self.merge_conv = nn.Sequential(
            nn.Conv2d(channels*5, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        s = self.short_field_conv(x)
        m = self.mid_field_conv(x)
        g = self.global_field_conv(x)
        c = self.chan_field_conv(x)
        merge = self.merge_conv(torch.cat((x, s, m, g, c), dim=1))
        return merge


class ConvNet_M2(nn.Module):
    def  __init__(self, channels=3, class_num=1):
        super(ConvNet_M2, self).__init__()
        channel_list = [64, 128, 256, 512, 512]
        self.class_num = class_num
        dilate_list = [(3, 6, 12, 18, 24),
                       (3, 6, 12, 18),
                       (3, 6, 12),
                       (1, 3, 6),
                       (1, 2, 3)]

        # dilate_list = [(3, 6, 12),
        #                (3, 6, 12),
        #                (3, 6, 12),
        #                (1, 3, 6),
        #                (1, 2, 3)]
        self.layer1 = nn.Sequential(
            conv3x3_bn_relu(channels, 64),
            MultiFieAtConv2d_M2(channel_list[0], dilate_list[0])
            # MFCModule(channel_list[0], num_layer_id=0)
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(64, 128),
            MultiFieAtConv2d_M2(channel_list[1], dilate_list[1]),
            # MFCModule(channel_list[1], num_layer_id=1),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(128, 256),
            MultiFieAtConv2d_M2(channel_list[2], dilate_list[2]),
            # MFCModule(channel_list[2], num_layer_id=2),
            conv3x3_bn_relu(256, 256),
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(256, 512),
            MultiFieAtConv2d_M2(channel_list[3], dilate_list[3]),
            # MFCModule(channel_list[3], num_layer_id=3),
            conv3x3_bn_relu(512, 512),
        )
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(512, 512),
            MultiFieAtConv2d_M2(channel_list[4], dilate_list[4]),
            # MFCModule(channel_list[4], num_layer_id=4),
            conv3x3_bn_relu(512, 512),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)
        return e1, e2, e3, e4, e5


class ConvNet_M4(nn.Module):
    def  __init__(self, channels=3, class_num=1):
        super(ConvNet_M4, self).__init__()
        channel_list = [64, 128, 256, 512, 512]
        self.class_num = class_num
        # dilate_list = [(3, 6, 12, 18, 24),
        #                (3, 6, 12, 18),
        #                (3, 6, 12),
        #                (1, 3, 6),
        #                (1, 2, 3)]

        dilate_list = [(3, 6, 12),
                       (3, 6, 12),
                       (3, 6, 12),
                       (1, 3, 6),
                       (1, 2, 3)]
        self.layer1 = nn.Sequential(
            conv3x3_bn_relu(channels, 64),
            MultiFieAtConv2d_M2(channel_list[0], dilate_list[0])
            # MFCModule(channel_list[0], num_layer_id=0)
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(64, 128),
            MultiFieAtConv2d_M2(channel_list[1], dilate_list[1]),
            # MFCModule(channel_list[1], num_layer_id=1),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(128, 256),
            MultiFieAtConv2d_M2(channel_list[2], dilate_list[2]),
            # MFCModule(channel_list[2], num_layer_id=2),
            conv3x3_bn_relu(256, 256),
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(256, 512),
            MultiFieAtConv2d_M2(channel_list[3], dilate_list[3]),
            # MFCModule(channel_list[3], num_layer_id=3),
            conv3x3_bn_relu(512, 512),
        )
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(512, 512),
            MultiFieAtConv2d_M2(channel_list[4], dilate_list[4]),
            # MFCModule(channel_list[4], num_layer_id=4),
            conv3x3_bn_relu(512, 512),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)
        return e1, e2, e3, e4, e5


class ConvNet_M3(nn.Module):
    def  __init__(self, channels=3, class_num=1):
        super(ConvNet_M3, self).__init__()
        channel_list = [64, 128, 256, 512, 512]
        self.class_num = class_num
        dilate_list = [(3, 6, 12),
                       (3, 6, 12),
                       (3, 6, 12),
                       (1, 3, 6),
                       (1, 2, 3)]
        k_list = [7, 5, 3, 3, 3]
        self.layer1 = nn.Sequential(
            conv3x3_bn_relu(channels, 64),
            MultiFieAtConv2d_M3(channel_list[0], k_list[0], dilate_list[0])
            # MFCModule(channel_list[0], num_layer_id=0)
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(64, 128),
            MultiFieAtConv2d_M3(channel_list[1], k_list[1], dilate_list[1]),
            # MFCModule(channel_list[1], num_layer_id=1),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(128, 256),
            MultiFieAtConv2d_M3(channel_list[2], k_list[2], dilate_list[2]),
            # MFCModule(channel_list[2], num_layer_id=2),
            conv3x3_bn_relu(256, 256),
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(256, 512),
            MultiFieAtConv2d_M3(channel_list[3], k_list[3], dilate_list[3]),
            # MFCModule(channel_list[3], num_layer_id=3),
            conv3x3_bn_relu(512, 512),
        )
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(512, 512),
            MultiFieAtConv2d_M3(channel_list[4], k_list[4], dilate_list[4]),
            # MFCModule(channel_list[4], num_layer_id=4),
            conv3x3_bn_relu(512, 512),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)
        return e1, e2, e3, e4, e5


class ConvNet_smlc(nn.Module):
    def  __init__(self, channels=3, class_num=1):
        super(ConvNet_smlc, self).__init__()
        channel_list = [64, 128, 256, 512, 512]
        self.class_num = class_num
        # dilate_list = [(3, 6, 12, 18, 24),
        #                (3, 6, 12, 18),
        #                (3, 6, 12),
        #                (1, 3, 6),
        #                (1, 2, 3)]

        dilate_list = [(3, 6, 12),
                       (3, 6, 12),
                       (3, 6, 12),
                       (1, 3, 6),
                       (1, 2, 3)]
        self.layer1 = nn.Sequential(
            conv3x3_bn_relu(channels, 64),
            MultiFieAtConv2d_SMLC(channel_list[0], dilate_list[0])
            # MFCModule(channel_list[0], num_layer_id=0)
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(64, 128),
            MultiFieAtConv2d_SMLC(channel_list[1], dilate_list[1]),
            # MFCModule(channel_list[1], num_layer_id=1),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(128, 256),
            MultiFieAtConv2d_SMLC(channel_list[2], dilate_list[2]),
            # MFCModule(channel_list[2], num_layer_id=2),
            conv3x3_bn_relu(256, 256),
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(256, 512),
            MultiFieAtConv2d_SMLC(channel_list[3], dilate_list[3]),
            # MFCModule(channel_list[3], num_layer_id=3),
            conv3x3_bn_relu(512, 512),
        )
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(512, 512),
            MultiFieAtConv2d_SMLC(channel_list[4], dilate_list[4]),
            # MFCModule(channel_list[4], num_layer_id=4),
            conv3x3_bn_relu(512, 512),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)
        return e1, e2, e3, e4, e5


class ConvNet_sm(nn.Module):
    def  __init__(self, channels=3, class_num=1):
        super(ConvNet_sm, self).__init__()
        channel_list = [64, 128, 256, 512, 512]
        self.class_num = class_num
        # dilate_list = [(3, 6, 12, 18, 24),
        #                (3, 6, 12, 18),
        #                (3, 6, 12),
        #                (1, 3, 6),
        #                (1, 2, 3)]

        self.layer1 = nn.Sequential(
            conv3x3_bn_relu(channels, 64),
            MultiFieAtConv2d_SM(channel_list[0])
            # MFCModule(channel_list[0], num_layer_id=0)
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(64, 128),
            MultiFieAtConv2d_SM(channel_list[1]),
            # MFCModule(channel_list[1], num_layer_id=1),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(128, 256),
            MultiFieAtConv2d_SM(channel_list[2]),
            # MFCModule(channel_list[2], num_layer_id=2),
            conv3x3_bn_relu(256, 256),
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(256, 512),
            MultiFieAtConv2d_SM(channel_list[3]),
            # MFCModule(channel_list[3], num_layer_id=3),
            conv3x3_bn_relu(512, 512),
        )
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(512, 512),
            MultiFieAtConv2d_SM(channel_list[4]),
            # MFCModule(channel_list[4], num_layer_id=4),
            conv3x3_bn_relu(512, 512),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)
        return e1, e2, e3, e4, e5


class ConvNet_sm2(nn.Module):
    def  __init__(self, channels=3, class_num=1):
        super(ConvNet_sm2, self).__init__()
        channel_list = [64, 128, 256, 512, 512]
        self.class_num = class_num
        # dilate_list = [(3, 6, 12, 18, 24),
        #                (3, 6, 12, 18),
        #                (3, 6, 12),
        #                (1, 3, 6),
        #                (1, 2, 3)]

        self.layer1 = nn.Sequential(
            conv3x3_bn_relu(channels, 64),
            MultiFieAtConv2d_SM2(channel_list[0])
            # MFCModule(channel_list[0], num_layer_id=0)
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(64, 128),
            MultiFieAtConv2d_SM2(channel_list[1]),
            # MFCModule(channel_list[1], num_layer_id=1),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(128, 256),
            MultiFieAtConv2d_SM2(channel_list[2]),
            # MFCModule(channel_list[2], num_layer_id=2),
            conv3x3_bn_relu(256, 256),
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(256, 512),
            MultiFieAtConv2d_SM2(channel_list[3]),
            # MFCModule(channel_list[3], num_layer_id=3),
            conv3x3_bn_relu(512, 512),
        )
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(512, 512),
            MultiFieAtConv2d_SM2(channel_list[4]),
            # MFCModule(channel_list[4], num_layer_id=4),
            conv3x3_bn_relu(512, 512),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)
        return e1, e2, e3, e4, e5


class ConvNet_mc(nn.Module):
    def  __init__(self, channels=3, class_num=1):
        super(ConvNet_mc, self).__init__()
        channel_list = [64, 128, 256, 512, 512]
        self.class_num = class_num
        # dilate_list = [(3, 6, 12, 18, 24),
        #                (3, 6, 12, 18),
        #                (3, 6, 12),
        #                (1, 3, 6),
        #                (1, 2, 3)]

        self.layer1 = nn.Sequential(
            conv3x3_bn_relu(channels, 64),
            MultiFieAtConv2d_MC(channel_list[0])
            # MFCModule(channel_list[0], num_layer_id=0)
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(64, 128),
            MultiFieAtConv2d_MC(channel_list[1]),
            # MFCModule(channel_list[1], num_layer_id=1),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(128, 256),
            MultiFieAtConv2d_MC(channel_list[2]),
            # MFCModule(channel_list[2], num_layer_id=2),
            conv3x3_bn_relu(256, 256),
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(256, 512),
            MultiFieAtConv2d_MC(channel_list[3]),
            # MFCModule(channel_list[3], num_layer_id=3),
            conv3x3_bn_relu(512, 512),
        )
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(512, 512),
            MultiFieAtConv2d_MC(channel_list[4]),
            # MFCModule(channel_list[4], num_layer_id=4),
            conv3x3_bn_relu(512, 512),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)
        return e1, e2, e3, e4, e5


class FCNM1Net(nn.Module):
    '''for FCN+MFA2'''

    def __init__(self):
        super(FCNM1Net, self).__init__()
        self.encoder = ConvNet_M2()
        self.decoder = DecFCN()

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        o1 = self.decoder(x, e3, e4, e5)
        return o1


class FCNM3Net(nn.Module):
    '''for FCN+MFA2'''

    def __init__(self):
        super(FCNM3Net, self).__init__()
        self.encoder = ConvNet_M3()
        self.decoder = DecFCN()

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        o1 = self.decoder(x, e3, e4, e5)
        return o1


class FCNM4Net(nn.Module):
    '''for FCN+MFA2'''

    def __init__(self):
        super(FCNM4Net, self).__init__()
        self.encoder = ConvNet_M4()
        self.decoder = DecFCN()

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        o1 = self.decoder(x, e3, e4, e5)
        return o1


class FCN_SMLC(nn.Module):
    '''for FCN+MFA2'''

    def __init__(self):
        super(FCN_SMLC, self).__init__()
        self.encoder = ConvNet_smlc()
        self.decoder = DecFCN()

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        o1 = self.decoder(x, e3, e4, e5)
        return o1


class FCN_SM(nn.Module):
    '''for FCN+MFA2'''

    def __init__(self):
        super(FCN_SM, self).__init__()
        self.encoder = ConvNet_sm()
        self.decoder = DecFCN()

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        o1 = self.decoder(x, e3, e4, e5)
        return o1


class FCN_SM2(nn.Module):
    '''for FCN+MFA2'''

    def __init__(self):
        super(FCN_SM2, self).__init__()
        self.encoder = ConvNet_sm2()
        self.decoder = DecFCN()

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        o1 = self.decoder(x, e3, e4, e5)
        return o1


class FCN_MC(nn.Module):
    '''for FCN+MFA2'''

    def __init__(self):
        super(FCN_MC, self).__init__()
        self.encoder = ConvNet_mc()
        self.decoder = DecFCN()

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        o1 = self.decoder(x, e3, e4, e5)
        return o1


class MFConvNet_Comb1(nn.Module):
    def __init__(self, channels=3, class_num=1):
        super(MFConvNet_Comb1, self).__init__()
        channel_list = [64, 128, 256, 512, 512]
        self.class_num = class_num
        self.layer1 = nn.Sequential(
            conv3x3_bn_relu(channels, 64),
            MultiFieAtConv2d(channel_list[0])
            # MFCModule(channel_list[0], num_layer_id=0)
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(64, 128),
            MultiFieAtConv2d(channel_list[1]),
            # MFCModule(channel_list[1], num_layer_id=1),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(128, 256),
            MultiFieAtConv2d(channel_list[2]),
            # MFCModule(channel_list[2], num_layer_id=2),
            conv3x3_bn_relu(256, 256),
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(256, 512),
            MultiFieAtConv2d(channel_list[3]),
            # MFCModule(channel_list[3], num_layer_id=3),
            conv3x3_bn_relu(512, 512),
        )
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(512, 512),
            MultiFieAtConv2d_Combin(channel_list[4]),
            # MFCModule(channel_list[4], num_layer_id=4),
            conv3x3_bn_relu(512, 512),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)
        return e1, e2, e3, e4, e5


class MFConvNet_Comb2(nn.Module):
    def __init__(self, channels=3, class_num=1):
        super(MFConvNet_Comb2, self).__init__()
        channel_list = [64, 128, 256, 512, 512]
        self.class_num = class_num
        self.layer1 = nn.Sequential(
            conv3x3_bn_relu(channels, 64),
            MultiFieAtConv2d(channel_list[0])
            # MFCModule(channel_list[0], num_layer_id=0)
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(64, 128),
            MultiFieAtConv2d(channel_list[1]),
            # MFCModule(channel_list[1], num_layer_id=1),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(128, 256),
            MultiFieAtConv2d(channel_list[2]),
            # MFCModule(channel_list[2], num_layer_id=2),
            conv3x3_bn_relu(256, 256),
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(256, 512),
            MultiFieAtConv2d_Combin(channel_list[3]),
            # MFCModule(channel_list[3], num_layer_id=3),
            conv3x3_bn_relu(512, 512),
        )
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_bn_relu(512, 512),
            MultiFieAtConv2d_Combin(channel_list[4]),
            # MFCModule(channel_list[4], num_layer_id=4),
            conv3x3_bn_relu(512, 512),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)
        return e1, e2, e3, e4, e5


class MFCNetComV1(nn.Module):
    '''for MFA2+GHL+MSP2'''

    def __init__(self):
        super(MFCNetComV1, self).__init__()
        self.encoder = MFConvNet_Comb1()
        self.decoder = DecoderNet()
        self.msp = MergeDecoder()

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        o1, o2, o3, o4, o5 = self.decoder(e1, e2, e3, e4, e5)
        res = self.msp(o1, o2, o3)
        return res, o1, o2, o3, o4, o5


class FCNComV1(nn.Module):
    '''for FCN+MFA2'''

    def __init__(self):
        super(FCNComV1, self).__init__()
        self.encoder = MFConvNet_Comb1()
        self.decoder = DecFCN()

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        o1 = self.decoder(x, e3, e4, e5)
        return o1


class FCNComV2(nn.Module):
    '''for FCN+MFA2'''

    def __init__(self):
        super(FCNComV2, self).__init__()
        self.encoder = MFConvNet_Comb2()
        self.decoder = DecFCN()

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        o1 = self.decoder(x, e3, e4, e5)
        return o1


if __name__ == '__main__':
    from thop import profile
    import torchsummary
    import torchscan
    from flopth import flopth
    band_num = 3
    class_num = 1

    # from torchscope import scope
    #
    # model = FCNM1Net()
    # scope(model, input_size=(3, 512, 512))
    # #
    # model = MFCNetComV1().eval().cuda()
    # sum_flop = flopth(model, in_size=[[3, 512, 512]])
    # print(sum_flop)
    # model = FCNM1Net()
    # from modelsummary import get_model_summary
    #
    # input_tensor = torch.rand(1, 3, 512, 512)
    # print(get_model_summary(model, input_tensor))

    # net.eval()
    #
    # if torch.cuda.is_available():
    #     net = net.cuda()
    #     x = x.cuda()


    # # torchscan.summary(model, (band_num, 512, 512), receptive_field=True, max_depth=0)
    input = torch.randn(1, 3, 512, 512).cuda()
    model = FCNComV2().eval().cuda()
    flops, params = profile(model, inputs=(input,))
    # # flops, params = profile(model, inputs=(1, band_num, 256, 256))
    # # model.cuda()
    # # torchsummary.summary(model, (band_num, 512, 512))
    print('flops(G): %.3f' % (flops / 1e+9))
    print('params(M): %.3f' % (params / 1e+6))
