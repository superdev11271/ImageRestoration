# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Scratch-mask detection network."""

import os

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from detection_models import networks as detection_networks

HERE = os.path.dirname(os.path.abspath(__file__))


class ScratchDetector:
    """Detection UNet: BGR image in, binary scratch mask out."""

    def __init__(self, GPU=0):
        self.GPU = GPU
        self.net = self._load_net()

    def _load_net(self):
        net = detection_networks.UNet(
            in_channels=1,
            out_channels=1,
            depth=4,
            conv_num=2,
            wf=6,
            padding=True,
            batch_norm=True,
            up_mode="upsample",
            with_tanh=False,
            sync_bn=True,
            antialiasing=True,
        )
        checkpoint = torch.load(
            os.path.join(HERE, "checkpoints", "detection", "FT_Epoch_latest.pt"), map_location="cpu"
        )
        net.load_state_dict(checkpoint["model_state"])
        if self.GPU >= 0:
            net.to(self.GPU)
        else:
            net.cpu()
        net.eval()
        return net

    @staticmethod
    def _scale_tensor(img_tensor, default_scale=256):
        """Scale the shorter side to default_scale, rounded to a multiple of 16."""
        _, _, w, h = img_tensor.shape
        if w < h:
            ow = default_scale
            oh = h / w * default_scale
        else:
            oh = default_scale
            ow = w / h * default_scale

        oh = int(round(oh / 16) * 16)
        ow = int(round(ow / 16) * 16)

        return F.interpolate(img_tensor, [ow, oh], mode="bilinear")

    @staticmethod
    def _transform(image):
        """Snap both sides to a multiple of 16, as the detection net expects."""
        h, w = image.shape[:2]
        nh = int(round(h / 16) * 16)
        nw = int(round(w / 16) * 16)
        if (nh, nw) == (h, w):
            return image
        return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)

    def detect(self, image):
        """BGR uint8 image in, binary 0/255 mask of the same size out."""
        h, w = image.shape[:2]

        gray = transforms.ToTensor()(cv2.cvtColor(self._transform(image), cv2.COLOR_BGR2GRAY))
        gray = transforms.Normalize([0.5], [0.5])(gray).unsqueeze(0)

        scaled = self._scale_tensor(gray)
        scaled = scaled.to(self.GPU) if self.GPU >= 0 else scaled.cpu()
        with torch.no_grad():
            P = torch.sigmoid(self.net(scaled))

        P = F.interpolate(P.data.cpu(), [h, w], mode="nearest")
        return (P[0, 0] >= 0.4).numpy().astype("uint8") * 255
