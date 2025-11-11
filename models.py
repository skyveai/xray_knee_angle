import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# CBAM: Channel + Spatial
# -------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(8, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(b, c, 1, 1)
        return x * attn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# -------------------------
# Residual bottleneck + CBAM
# -------------------------
class ResCBAM(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        mid = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.cbam = CBAM(out_ch)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.down is not None:
            identity = self.down(identity)

        out = out + identity
        out = self.cbam(out)
        out = self.relu(out)
        
        return out

# -------------------------
# Hourglass (recursive)
# -------------------------
class Hourglass(nn.Module):
    def __init__(self, depth: int, channels: int, block: nn.Module = ResCBAM):
        super().__init__()
        self.depth = depth
        self.up1 = block(channels, channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.low1 = block(channels, channels)
        self.low2 = Hourglass(depth - 1, channels, block) if depth > 1 else block(channels, channels)
        self.low3 = block(channels, channels)

    def forward(self, x):
        up1 = self.up1(x)                      # same res
        low = self.pool(x)                     # down
        low1 = self.low1(low)
        low2 = self.low2(low1)                 # recurse or block
        low3 = self.low3(low2)
        up2 = F.interpolate(low3, size=up1.shape[-2:], mode="bilinear", align_corners=False)
        return up1 + up2

# -------------------------
# Stacked Hourglass with CBAM head
# -------------------------
class StackedHourglassCBAM(nn.Module):
    """
    - Input: (B,1,512,512)
    - Internal pre-stem downsamples to 128x128, then applies hourglass.
    - Output: heatmaps (B, num_keypoints, 128, 128)
    """
    def __init__(self, num_keypoints=12, num_stacks=2, depth=4, channels=256, in_ch=1):
        super().__init__()
        self.num_stacks = num_stacks

        # Pre-stem: 512 -> 256 -> 128; also lift channels to 'channels'
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 512->256
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResCBAM(64, 128, stride=2),                                            # 256->128
            ResCBAM(128, channels, stride=1),
        )

        self.hourglasses = nn.ModuleList([Hourglass(depth, channels) for _ in range(num_stacks)])
        self.features = nn.ModuleList([nn.Sequential(
            ResCBAM(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ) for _ in range(num_stacks)])
        self.out_heads = nn.ModuleList([nn.Conv2d(channels, num_keypoints, kernel_size=1) for _ in range(num_stacks)])

        # intermediate supervision merging
        self.int_transforms = nn.ModuleList()
        self.int_skips = nn.ModuleList()
        for _ in range(num_stacks - 1):
            self.int_transforms.append(nn.Conv2d(channels, channels, kernel_size=1, bias=False))
            self.int_skips.append(nn.Conv2d(num_keypoints, channels, kernel_size=1, bias=False))

    def forward(self, x):
        x = self.stem(x)  # now 128x128 with 'channels'
        outputs = []
        feat = x
        for i in range(self.num_stacks):
            y = self.hourglasses[i](feat)
            y = self.features[i](y)
            
            out = self.out_heads[i](y)   # (B, J, 128, 128)
            outputs.append(out)
            if i < self.num_stacks - 1:
                feat = feat + self.int_transforms[i](y) + self.int_skips[i](out)
            
        return outputs #return outputs[-1] if self.training is False else outputs