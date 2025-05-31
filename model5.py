import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ResidualInputBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=0),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return F.relu(self.main(x) + self.skip(x))
    
class InputBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(InputBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(3),                  # avoid edge artifacts
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ResidualDownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualDownsampleBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.InstanceNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.conv(x)
        skip_out = self.skip(x)
        out = out + skip_out
        return out

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Feature refinement
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channels, pool_size=2):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))  #learnable weight
        self.pool = nn.AvgPool2d(pool_size)        #spatial size reduce
        self.upsample = nn.Upsample(scale_factor=pool_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        y = x
        x = self.pool(x)
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # B x N x C
        proj_key   = self.key(x).view(B, -1, H * W)                     # B x C x N
        energy = torch.bmm(proj_query, proj_key)                       # B x N x N
        attention = torch.softmax(energy, dim=-1)

        proj_value = self.value(x).view(B, C, -1)                      # B x C x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))       # B x C x N
        out = out.view(B, C, H, W)
        out = self.upsample(out)

        return self.gamma * out + y # same size

class DenseSkipBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers=4):
        super(DenseSkipBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.growth_rate = growth_rate

        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(layer_in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
                    nn.InstanceNorm2d(growth_rate),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)

class BottleneckWithAttention(nn.Module):
    def __init__(self, channels, num_blocks=6):
        super(BottleneckWithAttention, self).__init__()
        assert num_blocks >= 3, "Use at least 3 blocks for meaningful attention insertion"
        blocks = nn.ModuleList()
        for i in range(6):
            blocks.append(ResidualBlock(channels))
            if i in [1, 3, 5]:
                blocks.append(SelfAttention(channels))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class LRFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LRFE, self).__init__()
        
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        
        self.dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        
        self.fuse = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
                        if in_channels != out_channels else nn.Identity()
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_5x5 = self.conv5x5(x)
        x_dilated = self.dilated_conv(x)
        
        x_cat = torch.cat([x_5x5, x_dilated], dim=1)
        x_fused = self.fuse(x_cat)
        
        out = self.bn(x_fused + self.residual(x))
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(Encoder, self).__init__()
        self.input_block = ResidualInputBlock(in_channels, base_channels)
        self.down1 = DownsampleBlock(base_channels, base_channels * 2)      #size/2
        self.lrfe = LRFE(base_channels * 2, base_channels * 2)                      #size/2
        self.down2 = ResidualDownsampleBlock(base_channels * 2, base_channels * 4)  #size/4
        self.down3 = ResidualDownsampleBlock(base_channels * 4, base_channels * 8)  #size/4
        self.bottleneck = BottleneckWithAttention(base_channels * 8)
        self.denseskip = DenseSkipBlock(base_channels * 8, growth_rate=16, num_layers=3)
        self.reduce = nn.Conv2d(base_channels * 8 + 16 * 3, base_channels * 8, kernel_size=1)
    
    def forward(self, x):
        x = self.input_block(x)
        d1 = self.down1(x)
        d1 = self.lrfe(d1)
        d2 = self.down2(d1)
        #bottleneck_out = self.bottleneck(d2)
        d3 = self.down3(d2)
        bottleneck_out = self.bottleneck(d3)
        enhanced = self.denseskip(bottleneck_out)
        reduced = self.reduce(enhanced)
        return reduced, [d3, d2, d1]

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(ResidualDenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        curr_channels = in_channels
        for i in range(num_layers):
            self.layers.append(
                nn.Conv2d(curr_channels, growth_rate, kernel_size=3, padding=1)
            )
            curr_channels += growth_rate

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.local_fusion = nn.Conv2d(curr_channels, in_channels, kernel_size=1)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            with torch.no_grad():
                concatenated = torch.cat(features, 1)
            out = self.lrelu(layer(concatenated))

            #out = self.lrelu(layer(torch.cat(features, 1)))
            features.append(out)
        out = self.local_fusion(torch.cat(features, 1))
        return out + x  # Residual connection

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_attention=True):
        super(UpsampleBlock, self).__init__()
        self.use_attention = use_attention
        self.use_skip = skip_channels > 0

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.initial_refine = nn.Sequential(
            nn.Conv2d((out_channels + skip_channels) if self.use_skip else out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dense1 = DenseSkipBlock(out_channels, growth_rate=32, num_layers=4)
        self.dense2 = ResidualDenseBlock(out_channels + 128, growth_rate=32, num_layers=4)
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels + 128, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if self.use_attention:
            self.attention_block = SelfAttention(out_channels, pool_size=4)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if self.use_skip and skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.initial_refine(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.final_conv(x)
        if self.use_attention:
            x = self.attention_block(x)
        return x

class Decoder(nn.Module):
    def __init__(self, base_channels=64):
        super(Decoder, self).__init__()
        self.up1 = UpsampleBlock(base_channels * 8, base_channels * 4, skip_channels=512)  # 16 → 32
        self.up2 = UpsampleBlock(base_channels * 4, base_channels * 2, skip_channels=256)  # 32 → 64
        self.up3 = UpsampleBlock(base_channels * 2, base_channels, skip_channels=128)      # 64 → 128
        self.up4 = UpsampleBlock(base_channels, base_channels, skip_channels=0, use_attention=False)     # 128 → 256

        self.final = nn.Sequential(
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Output in [-1, 1] for GAN-style output
        )

    def forward(self, x, skips):
        x = self.up1(x, skips[0])  # d3
        x = self.up2(x, skips[1])  # d2
        x = self.up3(x, skips[2])  # d1
        x = self.up4(x)
        x = self.final(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(Discriminator, self).__init__()
        
        self.block1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)),  # Downsample
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels)
        )

        self.block2 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels * 2)
        )

        self.block3 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels * 4)
        )

        self.block4 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels * 8)
        )

        self.fc = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=2, padding=1)),  # Output probability
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.fc(out)
        #out = torch.flatten(out, 1).mean(1, keepdim=True)
        return out.view(-1, 1)
    
    def get_intermediate_features(self, x):
        feats = []
        out = self.block1(x)
        feats.append(out)
        out = self.block2(out)
        feats.append(out)
        out = self.block3(out)
        feats.append(out)
        out = self.block4(out)
        feats.append(out)
        return feats

class Generator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(Generator, self).__init__()
        self.encoder = Encoder(in_channels, base_channels)
        self.decoder = Decoder(base_channels)

    def forward(self, x):
        bottleneck_out, skips = self.encoder(x)
        output = self.decoder(bottleneck_out, skips)
        return output

class GAN(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(GAN, self).__init__()
        self.generator = Generator(in_channels, base_channels)
        self.discriminator = Discriminator(in_channels=in_channels, base_channels=base_channels)

    def forward(self, x):
        generated_image = self.generator(x)
        discriminator_output = self.discriminator(generated_image)
        return generated_image, discriminator_output