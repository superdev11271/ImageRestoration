# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Single-image restoration inference.

Loads the scratch-detection and restoration networks once and restores an
image held in memory, instead of shelling out to detection.py / test.py per
directory the way run.py does.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from detection import data_transforms as detection_data_transforms, scale_tensor
from detection_models import networks as detection_networks
from models.mapping_model import Pix2PixHDModel_Mapping
from options.test_options import TestOptions
from test import data_transforms, irregular_hole_synthesize, parameter_set

HERE = os.path.dirname(os.path.abspath(__file__))


class ImageRestorer:
    """Old-photo restoration for a single BGR image."""

    def __init__(self, with_scratch=False, HR=False, GPU=0):
        self.with_scratch = with_scratch
        self.HR = HR
        self.GPU = GPU

        self.opt = self._build_opt()

        self.detection_net = self._load_detection_net() if with_scratch else None

        self.model = Pix2PixHDModel_Mapping()
        self.model.initialize(self.opt)
        self.model.eval()

        self.img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.mask_transform = transforms.ToTensor()

    ## ---------------------------------------------------------------- setup

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

    def _load_detection_net(self):
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

    ## ------------------------------------------------------------ pipeline

    def preprocess(self, image):
        """BGR uint8 image -> (input tensor, mask tensor) ready for the model."""
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not self.with_scratch:
            pil = data_transforms(pil, scale=False)
            input_tensor = self.img_transform(pil).unsqueeze(0)
            return input_tensor, torch.zeros_like(input_tensor)

        pil = detection_data_transforms(pil, "full_size")
        mask = self._detect_scratch(pil)
        if self.opt.mask_dilation != 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = Image.fromarray(
                cv2.dilate(np.array(mask), kernel, iterations=self.opt.mask_dilation).astype("uint8")
            )

        holed = irregular_hole_synthesize(pil, mask)
        input_tensor = self.img_transform(holed).unsqueeze(0)
        mask_tensor = self.mask_transform(mask)[:1, :, :].unsqueeze(0)  # single channel
        return input_tensor, mask_tensor

    def _detect_scratch(self, pil_image):
        """Run the detection UNet, returning a binary 0/255 RGB mask."""
        gray = transforms.ToTensor()(pil_image.convert("L"))
        gray = transforms.Normalize([0.5], [0.5])(gray).unsqueeze(0)
        _, _, ow, oh = gray.shape

        scaled = scale_tensor(gray)
        scaled = scaled.to(self.GPU) if self.GPU >= 0 else scaled.cpu()
        with torch.no_grad():
            P = torch.sigmoid(self.detection_net(scaled))

        P = F.interpolate(P.data.cpu(), [ow, oh], mode="nearest")
        mask = (P[0, 0] >= 0.4).numpy().astype("uint8") * 255
        return Image.fromarray(mask).convert("RGB")

    def inference(self, image):
        """BGR uint8 image in, restored BGR uint8 image out."""
        input_tensor, mask_tensor = self.preprocess(image)
        with torch.no_grad():
            generated = self.model.inference(input_tensor, mask_tensor)
        return self.postprocess(generated)

    def postprocess(self, generated):
        """Model output -> BGR uint8, matching save_image(..., normalize=True)."""
        img = ((generated.data.cpu() + 1.0) / 2.0).squeeze(0)
        low, high = float(img.min()), float(img.max())
        img = img.clamp_(min=low, max=high).sub_(low).div_(max(high - low, 1e-5))
        array = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).numpy().astype("uint8")
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./test_images/a.png", help="Input image")
    parser.add_argument("--GPU", type=int, default=0, help="GPU id, -1 for CPU")
    parser.add_argument("--with_scratch", action="store_true")
    parser.add_argument("--HR", action="store_true")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        sys.exit("Could not read image: %s" % args.input)

    restorer = ImageRestorer(with_scratch=args.with_scratch, HR=args.HR, GPU=args.GPU)
    result = restorer.inference(image)

    root, ext = os.path.splitext(args.input)
    output_path = root + "_out" + ext
    cv2.imwrite(output_path, result)
    print("Saved %s" % output_path)
