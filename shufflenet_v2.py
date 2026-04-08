import torch
import torch.nn as nn

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]
    
    
class FusedIB2(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, ):
        super().__init__()
        self.use_res = stride == 1 and in_chs == out_chs

        # 1. Fused Layer: 3x3 Standard Conv (Expands and processes spatially)
        self.fused = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chs, out_chs, 1, stride=1, bias=False),
        )

    def forward(self, x):
        out = self.fused(x)
        
        if self.use_res:
            return out + x
        return out    
    
    
class FusedIB(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, exp_ratio=4):
        super().__init__()
        mid_chs = int(in_chs * exp_ratio)
        self.use_res = stride == 1 and in_chs == out_chs

        # 1. Fused Layer: 3x3 Standard Conv (Expands and processes spatially)
        self.fused = nn.Sequential(
            nn.Conv2d(in_chs, mid_chs, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_chs),
            nn.ReLU6(inplace=True)
        )

        # 2. Projection Layer: 1x1 Conv (Compresses back to output size)
        # Note: No activation function here (Linear Bottleneck)
        self.project = nn.Sequential(
            nn.Conv2d(mid_chs, out_chs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chs)
        )

    def forward(self, x):
        out = self.fused(x)
        out = self.project(out)
        
        if self.use_res:
            return out + x
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, input_size=224, n_class=1000, model_size='0.75x', fused_ib = False):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '0.75x':
            self.stage_out_channels = [-1, 24, 108, 216, 432, 1024]
        elif model_size == '0.90x':
            self.stage_out_channels = [-1, 24, 108, 216, 448, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )
        
        
        self.fused_or_max = FusedIB2(input_channel, input_channel, stride=2) if fused_ib else nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            for i in range(numrepeat):
                if i == 0:
                    self.features.append(ShuffleV2Block(input_channel, output_channel, 
                                                mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    self.features.append(ShuffleV2Block(input_channel // 2, output_channel, 
                                                mid_channels=output_channel // 2, ksize=3, stride=1))

                input_channel = output_channel
                
        self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1]),
            nn.ReLU(inplace=True)
        )
        self.globalpool = nn.AvgPool2d(7)
        if self.model_size == '2.0x':
            self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class, bias=False))

    def forward(self, x):
        x = self.first_conv(x)
        x = self.fused_or_max(x)
        x = self.features(x)
        x = self.conv_last(x)

        x = self.globalpool(x)
        if self.model_size == '2.0x':
            x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    model = ShuffleNetV2(n_class=10, model_size="0.80x")
    # number of parameters in shufflenet_v2
    total_params_shuffle = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in ShuffleNetV2_080: {total_params_shuffle}")