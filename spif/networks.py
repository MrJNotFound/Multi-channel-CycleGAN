"""SPIF V2 forward/reverse generators and their shared building blocks."""

import functools

import torch
from torch import nn
import torch.nn.functional as F


def instance_norm():
    """Use the normalization of the SPIF experiments."""
    return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)


def initialize_weights(network):
    """Initialize convolution weights with N(0, 0.02), as in the original model."""
    for layer in network.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(layer.weight, 0.0, 0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def forward_generator(ngf=64, n_blocks=9):
    """Build the fixed two-channel BF/AF to RGB H&E architecture."""
    return SPIFV2Generator(2, 3, ngf=ngf, n_blocks=n_blocks, norm_layer=instance_norm())


class PatchDiscriminator(nn.Module):
    """The three-layer 70x70 PatchGAN used by the SPIF training objective."""

    def __init__(self, input_nc, ndf=64):
        super().__init__()
        norm = instance_norm()
        layers = [nn.Conv2d(input_nc, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        previous = 1
        for level in (1, 2):
            current = 2 ** level
            layers += [nn.Conv2d(ndf * previous, ndf * current, 4, 2, 1, bias=True),
                       norm(ndf * current), nn.LeakyReLU(0.2, True)]
            previous = current
        layers += [nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=True),
                   norm(ndf * 8), nn.LeakyReLU(0.2, True),
                   nn.Conv2d(ndf * 8, 1, 4, 1, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, image):
        return self.model(image)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


def _spif_uses_bias(norm_layer):
    """Match the bias convention used by the CycleGAN ResNet generators."""
    if isinstance(norm_layer, functools.partial):
        return norm_layer.func == nn.InstanceNorm2d
    return norm_layer == nn.InstanceNorm2d


def _spif_downsample(in_channels, out_channels, norm_layer, use_bias):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,
                  padding=1, bias=use_bias),
        norm_layer(out_channels),
        nn.ReLU(inplace=True),
    )


def _spif_upsample(in_channels, out_channels, norm_layer, use_bias):
    """Resize-convolution upsampling avoids transposed-convolution checkerboards."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0,
                  bias=use_bias),
        norm_layer(out_channels),
        nn.ReLU(inplace=True),
    )


class SPIFCompetitiveFusion(nn.Module):
    """Spatial-channel competitive fusion for two aligned modalities.

    Channel logits capture patch-level modality reliability while spatial
    logits allow that preference to change across tissue regions.  The two
    modality weights are complementary, so an unreliable modality can be
    suppressed instead of always bypassing the attention through a raw sum.
    A zero-initialized residual interaction path lets the module start from a
    stable convex mixture and learn cross-modal corrections gradually.
    """

    def __init__(self, channels, norm_layer=nn.BatchNorm2d, reduction=4):
        super().__init__()
        use_bias = _spif_uses_bias(norm_layer)
        channel_hidden = max(channels // reduction, 8)
        spatial_hidden = max(channels // 8, 8)

        self.channels = channels
        self.channel_logits = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channel_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_hidden, channels * 2, kernel_size=1),
        )
        self.spatial_logits = nn.Sequential(
            nn.Conv2d(channels * 2, spatial_hidden, kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_hidden, 2, kernel_size=1),
        )
        self.interaction = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=use_bias),
            norm_layer(channels),
        )
        self.output = nn.Sequential(
            norm_layer(channels),
            nn.ReLU(inplace=True),
        )
        self.interaction_scale = nn.Parameter(torch.zeros(1))

    def compute_modality_weights(self, feat_bf, feat_af):
        """Return complementary BF/AF weights with shape (B, C, H, W)."""
        if feat_bf.shape != feat_af.shape:
            raise ValueError(
                "BF and AF feature maps must have identical shapes, got "
                f"{tuple(feat_bf.shape)} and {tuple(feat_af.shape)}."
            )
        joint = torch.cat([feat_bf, feat_af], dim=1)
        batch, _, height, width = feat_bf.shape
        channel_logits = self.channel_logits(joint).view(
            batch, 2, self.channels, 1, 1
        )
        spatial_logits = self.spatial_logits(joint).view(
            batch, 2, 1, height, width
        )

        # A two-class softmax can be evaluated as one sigmoid of the logit
        # difference, avoiding a materialized (B, 2, C, H, W) tensor.
        logit_delta = (
            channel_logits[:, 0] - channel_logits[:, 1]
            + spatial_logits[:, 0] - spatial_logits[:, 1]
        )
        weight_bf = torch.sigmoid(logit_delta)
        return weight_bf, 1.0 - weight_bf

    def forward(self, feat_bf, feat_af):
        weight_bf, weight_af = self.compute_modality_weights(feat_bf, feat_af)
        selected = weight_bf * feat_bf + weight_af * feat_af
        interaction_features = torch.cat(
            [feat_bf * feat_af, torch.abs(feat_bf - feat_af)], dim=1
        )
        correction = self.interaction(interaction_features)
        fused = selected + torch.tanh(self.interaction_scale) * correction
        return self.output(fused)


class SPIFV2Generator(nn.Module):
    """Single-stage competitive-fusion generator for BF/AF translation.

    The modalities use separate encoders through H/2, where joint spatial and
    channel competition produces one shared representation.  A shared
    downsampling stage then feeds the ResNet body.  The H/2 fused representation
    is also reused as a decoder skip so fine boundaries need not pass only
    through the bottleneck.
    """

    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type="reflect"):
        assert input_nc == 2, (
            f"SPIFV2Generator requires input_nc=2, got {input_nc}"
        )
        assert n_blocks >= 0
        super().__init__()
        use_bias = _spif_uses_bias(norm_layer)

        def modality_head():
            return nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(1, ngf, kernel_size=7, padding=0, bias=use_bias),
                norm_layer(ngf),
                nn.ReLU(inplace=True),
                *_spif_downsample(ngf, ngf * 2, norm_layer, use_bias),
            )

        self.head_bf = modality_head()
        self.head_af = modality_head()
        self.fusion_half = SPIFCompetitiveFusion(
            ngf * 2, norm_layer=norm_layer
        )

        # After the single H/2 competition, later encoding is shared.  This
        # avoids constructing two highly correlated branches from an already
        # fused feature map and then immediately fusing them again.
        self.down_shared = _spif_downsample(
            ngf * 2, ngf * 4, norm_layer, use_bias
        )

        body = []
        for _ in range(n_blocks):
            body.append(
                ResnetBlock(
                    ngf * 4, padding_type=padding_type,
                    norm_layer=norm_layer, use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            )
        self.body = nn.Sequential(*body)

        self.up_to_half = _spif_upsample(
            ngf * 4, ngf * 2, norm_layer, use_bias
        )
        self.skip_merge = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(inplace=True),
        )
        self.up_to_full = _spif_upsample(
            ngf * 2, ngf, norm_layer, use_bias
        )
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):
        feat_bf_half = self.head_bf(x[:, 0:1])
        feat_af_half = self.head_af(x[:, 1:2])
        fused_half = self.fusion_half(feat_bf_half, feat_af_half)
        shared_quarter = self.down_shared(fused_half)

        decoded_half = self.up_to_half(self.body(shared_quarter))
        if decoded_half.shape[-2:] != fused_half.shape[-2:]:
            decoded_half = F.interpolate(
                decoded_half, size=fused_half.shape[-2:], mode="bilinear",
                align_corners=False,
            )
        decoded_half = self.skip_merge(
            torch.cat([decoded_half, fused_half], dim=1)
        )
        return self.output(self.up_to_full(decoded_half))


class SPIFDualV2Generator(nn.Module):
    """Reverse generator with shared geometry and modality-specific texture.

    The H&E encoder and ResNet body are shared.  Decoding is shared up to H/2,
    then BF and AF use independent high-resolution decoders rather than only
    independent final convolutions.
    """

    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type="reflect"):
        assert output_nc == 2, (
            f"SPIFDualV2Generator requires output_nc=2, got {output_nc}"
        )
        assert n_blocks >= 0
        super().__init__()
        use_bias = _spif_uses_bias(norm_layer)

        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                      bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(inplace=True),
        )
        self.down_to_half = _spif_downsample(
            ngf, ngf * 2, norm_layer, use_bias
        )
        self.down_to_quarter = _spif_downsample(
            ngf * 2, ngf * 4, norm_layer, use_bias
        )

        body = []
        for _ in range(n_blocks):
            body.append(
                ResnetBlock(
                    ngf * 4, padding_type=padding_type,
                    norm_layer=norm_layer, use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            )
        self.body = nn.Sequential(*body)
        self.up_shared = _spif_upsample(
            ngf * 4, ngf * 2, norm_layer, use_bias
        )
        self.skip_merge = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(inplace=True),
        )

        def modality_decoder():
            return nn.Sequential(
                *_spif_upsample(ngf * 2, ngf, norm_layer, use_bias),
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, 1, kernel_size=7, padding=0),
                nn.Tanh(),
            )

        self.decoder_bf = modality_decoder()
        self.decoder_af = modality_decoder()

    def forward(self, x):
        feat_full = self.stem(x)
        feat_half = self.down_to_half(feat_full)
        feat_quarter = self.down_to_quarter(feat_half)
        decoded_half = self.up_shared(self.body(feat_quarter))
        if decoded_half.shape[-2:] != feat_half.shape[-2:]:
            decoded_half = F.interpolate(
                decoded_half, size=feat_half.shape[-2:], mode="bilinear",
                align_corners=False,
            )
        decoded_half = self.skip_merge(
            torch.cat([decoded_half, feat_half], dim=1)
        )
        bf = self.decoder_bf(decoded_half)
        af = self.decoder_af(decoded_half)
        return torch.cat([bf, af], dim=1)

