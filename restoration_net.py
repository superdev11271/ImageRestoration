# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Old-photo restoration (Pix2PixHD mapping) network."""

import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from models.mapping_model import Pix2PixHDModel_Mapping
from options.test_options import TestOptions
from test import parameter_set

HERE = os.path.dirname(os.path.abspath(__file__))


class RestorationNet:
    """Pix2PixHD mapping model: BGR image (+ scratch mask) in, restored BGR out."""

    def __init__(self, with_scratch=False, HR=False, GPU=0):
        self.with_scratch = with_scratch
        self.HR = HR
        self.GPU = GPU

        self.opt = self._build_opt()
        self.model = Pix2PixHDModel_Mapping()
        self.model.initialize(self.opt)
        self.model.eval()

        self.img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.mask_transform = transforms.ToTensor()

    def _build_opt(self):
        """Build the same option namespace run.py would produce on the CLI."""
        argv = ["--gpu_ids", str(self.GPU)]
        if self.with_scratch:
            argv.append("--Scratch_and_Quality_restore")
            if self.HR:
                argv.append("--HR")
        else:
            argv += ["--test_mode", "Full", "--Quality_restore"]

        saved_argv = sys.argv
        sys.argv = [saved_argv[0]] + argv
        try:
            opt = TestOptions().parse(save=False)
        finally:
            sys.argv = saved_argv

        parameter_set(opt)

        # parameter_set uses paths relative to the cwd; anchor them to this file
        opt.checkpoints_dir = os.path.join(HERE, "checkpoints", "restoration")
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, os.path.basename(opt.load_pretrainA))
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, os.path.basename(opt.load_pretrainB))
        return opt

    def restore(self, image, mask=None):
        """BGR uint8 image and optional 0/255 mask in, restored BGR uint8 out."""
        input_tensor, mask_tensor = self.preprocess(image, mask)
        with torch.no_grad():
            generated = self.model.inference(input_tensor, mask_tensor)
        return self.postprocess(generated)

    @staticmethod
    def _transform(image):
        """Snap both sides to a multiple of 4, as the mapping model expects."""
        h, w = image.shape[:2]
        nh = int(round(h / 4) * 4)
        nw = int(round(w / 4) * 4)
        if (nh, nw) == (h, w):
            return image
        return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    def preprocess(self, image, mask=None):
        """BGR image (+ mask) -> (input tensor, mask tensor) ready for the model."""
        rgb = self._transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if mask is None:
            input_tensor = self.img_transform(rgb).unsqueeze(0)
            return input_tensor, torch.zeros_like(input_tensor)

        h, w = rgb.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if self.opt.mask_dilation != 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=self.opt.mask_dilation)

        input_tensor = self.img_transform(self._synthesize_hole(rgb, mask)).unsqueeze(0)
        mask_tensor = self.mask_transform(mask).unsqueeze(0)
        return input_tensor, mask_tensor

    @staticmethod
    def _synthesize_hole(image, mask):
        """Paint the masked pixels white, the holes the mapping model fills in."""
        holes = (mask / 255)[:, :, None]
        return (image * (1 - holes) + holes * 255).astype("uint8")

    def postprocess(self, generated):
        """Model output -> BGR uint8, matching save_image(..., normalize=True)."""
        img = ((generated.data.cpu() + 1.0) / 2.0).squeeze(0)
        low, high = float(img.min()), float(img.max())
        img = img.clamp_(min=low, max=high).sub_(low).div_(max(high - low, 1e-5))
        array = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).numpy().astype("uint8")
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
