# coding: utf-8

# Standard imports

# External imports
import torch
import torch.nn as nn
import timm
import torchvision
import torchvision.models.segmentation as segmentation


class DeepLabV3(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        # vvvvvvvvv
        # CODE HERE
        self.model = None
        # ^^^^^^^^^

    def forward(self, x):
        output = self.model(x)
        return output["out"]


def conv_relu_bn(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
        nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]


class TimmEncoder(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.model = timm.create_model(
            model_name="resnet18", in_chans=cin, pretrained=True
        )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)

        x = self.model.maxpool(x)

        f1 = self.model.layer1(x)
        f2 = self.model.layer2(f1)
        f3 = self.model.layer3(f2)
        f4 = self.model.layer4(f3)

        return f4, [f1, f2, f3]


class DecoderBlock(nn.Module):
    def __init__(self, cin):
        super().__init__()
        # vvvvvvvvv
        # CODE HERE
        self.conv1 = None
        self.up_conv = None
        self.conv2 = None
        # ^^^^^^^^^

    def forward(self, x, f_encoder):
        # On passe à travers les premières couches convolutives et upsampling
        # # vvvvvvvvv
        # # CODE HERE
        x = None
        # # ^^^^^^^^^

        # On concatène les features de l'encoder
        # x et f_encoder sont (B, C, H, W)
        # # vvvvvvvvv
        # # CODE HERE
        x = None
        # # ^^^^^^^^^

        # On applique la dernière convolution
        # # vvvvvvvvv
        # # CODE HERE
        out = None
        # # ^^^^^^^^^

        return out


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        cbase = 64
        self.b1 = DecoderBlock(cin=8 * cbase)
        self.b2 = DecoderBlock(cin=4 * cbase)
        self.b3 = DecoderBlock(cin=2 * cbase)  # cout=cbase

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2), *conv_relu_bn(cbase, cbase)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2), *conv_relu_bn(cbase, cbase)
        )
        self.last_conv = nn.Sequential(*conv_relu_bn(cbase, num_classes))

    def forward(self, f4, f_encoder):
        [f1, f2, f3] = f_encoder

        x1 = self.b1(f4, f3)
        x2 = self.b2(x1, f2)
        x3 = self.b3(x2, f1)

        out = self.last_conv(self.up2(self.up1(x3)))

        return out


class UNet(nn.Module):
    def __init__(self, cin, num_classes):
        super().__init__()
        self.encoder = TimmEncoder(cin)
        self.decoder = Decoder(num_classes)

    def forward(self, X):
        out, features = self.encoder(X)
        prediction = self.decoder(out, features)
        return prediction







def test_unet():
    X = torch.zeros((1, 3, 256, 256))
    num_classes = 21
    cin = 3
    model = UNet(cin, num_classes)
    model.eval()
    y = model(X)
    print(f"Output shape : {y.shape}")


def test_timm():
    x = torch.zeros((1, 3, 256, 256))
    model = timm.create_model(model_name="resnet18", pretrained=True)
    model.eval()

    x = model.conv1(x)
    x = model.bn1(x)
    x = model.act1(x)
    x = model.maxpool(x)

    f1 = model.layer1(x)
    f2 = model.layer2(f1)
    f3 = model.layer3(f2)
    f4 = model.layer4(f3)



if __name__ == "__main__":
    # test_timm()
    test_unet()
