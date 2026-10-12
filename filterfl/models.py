"""Model definitions — independently written.

ResNet8 here reproduces the *architecture* documented in CODE_NOTES.md (wide
ResNet-8: stage widths 128/256/256*2 i.e. 128->256->512, BasicBlock x[1,1,1],
avgpool+linear head) so capacity matches the reference for a fair comparison,
but the code is written from scratch (different structure/names), not copied
from reference/filterfl/models/ResNet8.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.n1 = nn.BatchNorm2d(c_out)
        self.c2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.n2 = nn.BatchNorm2d(c_out)

        self.skip = None
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x):
        out = F.relu(self.n1(self.c1(x)))
        out = self.n2(self.c2(out))
        identity = x if self.skip is None else self.skip(x)
        return F.relu(out + identity)


class ResNet8(nn.Module):
    """Wide ResNet-8: stem(128) -> stage(128) -> stage(256, /2) -> stage(512, /2) -> avgpool -> fc."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.stem_conv = nn.Conv2d(in_channels, 128, 3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(128)
        self.stage1 = ResBlock(128, 128, stride=1)
        self.stage2 = ResBlock(128, 256, stride=2)
        self.stage3 = ResBlock(256, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.stem_bn(self.stem_conv(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out).flatten(1)
        return self.fc(out)


class ResBlockNoNorm(nn.Module):
    """ResBlock with BatchNorm2d replaced by nn.Identity() — for the Stage 3
    architecture-dependence ablation (CODE_NOTES.md): does detection survive
    on a model with nothing to renormalize against a trigger batch?"""
    expansion = 1

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=True)
        self.n1 = nn.Identity()
        self.c2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=True)
        self.n2 = nn.Identity()

        self.skip = None
        if stride != 1 or c_in != c_out:
            self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=True)

    def forward(self, x):
        out = F.relu(self.n1(self.c1(x)))
        out = self.n2(self.c2(out))
        identity = x if self.skip is None else self.skip(x)
        return F.relu(out + identity)


class ResNet8NoNorm(nn.Module):
    """Same topology as ResNet8 (128->256->512, BasicBlock x[1,1,1]) with
    every BatchNorm2d replaced by nn.Identity() — no normalization layer of
    any kind, so there is nothing for train/eval mode to change."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.stem_conv = nn.Conv2d(in_channels, 128, 3, stride=1, padding=1, bias=True)
        self.stem_bn = nn.Identity()
        self.stage1 = ResBlockNoNorm(128, 128, stride=1)
        self.stage2 = ResBlockNoNorm(128, 256, stride=2)
        self.stage3 = ResBlockNoNorm(256, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.stem_bn(self.stem_conv(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out).flatten(1)
        return self.fc(out)


class ResBlockGroupNorm(nn.Module):
    """ResBlock with BatchNorm2d replaced by GroupNorm — trains comparably
    well (unlike ResBlockNoNorm) but normalizes per-sample, so it has no
    train/eval mode distinction and is immune to batch composition. Isolates
    "no batch-dependent renormalization" from "undertrained/malformed net"
    in the CODE_NOTES.md architecture-dependence ablation."""
    expansion = 1

    def __init__(self, c_in, c_out, stride=1, groups=32):
        super().__init__()
        self.c1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.n1 = nn.GroupNorm(min(groups, c_out), c_out)
        self.c2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.n2 = nn.GroupNorm(min(groups, c_out), c_out)

        self.skip = None
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.GroupNorm(min(groups, c_out), c_out),
            )

    def forward(self, x):
        out = F.relu(self.n1(self.c1(x)))
        out = self.n2(self.c2(out))
        identity = x if self.skip is None else self.skip(x)
        return F.relu(out + identity)


class ResNet8GroupNorm(nn.Module):
    """Same topology as ResNet8 (128->256->512, BasicBlock x[1,1,1]) with
    every BatchNorm2d replaced by GroupNorm(32, C)."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.stem_conv = nn.Conv2d(in_channels, 128, 3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.GroupNorm(32, 128)
        self.stage1 = ResBlockGroupNorm(128, 128, stride=1)
        self.stage2 = ResBlockGroupNorm(128, 256, stride=2)
        self.stage3 = ResBlockGroupNorm(256, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.stem_bn(self.stem_conv(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out).flatten(1)
        return self.fc(out)


class CNNMnist(nn.Module):
    """Small CNN for MNIST smoke tests (cheap, not in the reference repo at all)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 5, padding=2)
        self.c2 = nn.Conv2d(16, 32, 5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class GeneratorCIFAR(nn.Module):
    """CGAN generator used by Stage 1/2 — image_num x latent_dim -> image_num x out_dim x 32 x 32."""

    def __init__(self, out_dim: int = 3, latent_dim: int = 100):
        super().__init__()
        self.init_size = 32 // 4
        self.fc = nn.Linear(latent_dim, 128 * self.init_size ** 2)
        self.body = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, out_dim, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.fc(z).view(z.size(0), 128, self.init_size, self.init_size)
        return self.body(out)


def gan_weight_init(m):
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm2d" in name:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def build_model(cfg_type: str, num_classes: int = 10) -> nn.Module:
    if cfg_type == "MNIST":
        return CNNMnist(num_classes=num_classes)
    return ResNet8(num_classes=num_classes)
