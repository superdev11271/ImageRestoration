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
import torchvision.transforms.functional as TF

from detection_models import networks as detection_networks
from models.mapping_model import Pix2PixHDModel_Mapping
from options.test_options import TestOptions

HERE = os.path.dirname(os.path.abspath(__file__))


def resize_to_multiple(image, base=16, interpolation=cv2.INTER_CUBIC):
    """Round both sides of an image array to a multiple of `base`.

    Both networks only need their input divisible by a power of two (4 for the
    restoration generators, 16 for the detection UNet), so rounding to 16
    satisfies both.
    """
    oh, ow = image.shape[:2]
    w = int(round(ow / base) * base)
    h = int(round(oh / base) * base)
    if (w, h) == (ow, oh):
        return image
    return cv2.resize(image, (w, h), interpolation=interpolation)


def scale_short_side(img_tensor, default_scale=256):
    """Resize so the shorter side is `default_scale`, keeping the aspect ratio.

    The long side is integer arithmetic on the input's shape, which survives
    ONNX export as int64 ops, so the exported graph handles any input size.
    The `if` cannot: torch.export resolves it from the sample input, so an
    exported graph assumes that sample's orientation (see export_onnx.py).
    Keeping it as a branch is deliberate - hiding it in torch.cond makes the
    result's size opaque and the detection UNet then fails to export.

    The result is a multiple of 16, which the UNet's four downsampling steps
    need.
    """
    h, w = img_tensor.shape[2], img_tensor.shape[3]
    unit = default_scale // 16  # long side in units of 16: round(unit * long / short)
    if h <= w:
        size = [default_scale, (2 * unit * w + h) // (2 * h) * 16]
    else:
        size = [(2 * unit * h + w) // (2 * w) * 16, default_scale]
    return F.interpolate(img_tensor, size, mode="bilinear")


def dilate_mask(mask, iterations):
    """Binary dilation of a mask tensor.

    `iterations` passes of a 3x3 rectangular kernel grow the mask by the same
    amount as one (2 * iterations + 1) square, which is a single max-pool.
    """
    size = 2 * iterations + 1
    return F.max_pool2d(mask, size, stride=1, padding=iterations)


class ImageRestorer:
    """Old-photo restoration for a single BGR image."""

    def __init__(self, HR=False, device="cuda"):
        self.HR = HR
        self.device = device

        self.opt = self._build_opt()

        self.detection_net = self._load_detection_net()

        self.model = Pix2PixHDModel_Mapping()
        self.model.initialize(self.opt)
        self.model.eval()

    ## ---------------------------------------------------------------- setup

    def _build_opt(self):
        """Build the same option namespace run.py would produce on the CLI."""
        saved_argv = sys.argv
        sys.argv = [saved_argv[0], "--gpu_ids", "0" if self.device == "cuda" else "-1"]
        try:
            opt = TestOptions().parse(save=False)
        finally:
            sys.argv = saved_argv

        # same settings test.py's parameter_set() applies
        opt.serial_batches = True
        opt.no_flip = True
        opt.label_nc = 0
        opt.n_downsample_global = 3
        opt.mc = 64
        opt.k_size = 4
        opt.start_r = 1
        opt.mapping_n_block = 6
        opt.map_mc = 512
        opt.no_instance = True
        opt.checkpoints_dir = os.path.join(HERE, "checkpoints", "restoration")
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")

        opt.NL_res = True
        opt.use_SN = True
        opt.correlation_renormalize = True
        opt.NL_use_mask = True
        opt.NL_fusion_method = "combine"
        opt.non_local = "Setting_42"
        opt.name = "mapping_scratch"
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_scratch")
        if self.HR:
            opt.mapping_exp = 1
            opt.inference_optimize = True
            opt.mask_dilation = 3
            opt.name = "mapping_Patch_Attention"

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
        net.to(self.device)
        net.eval()
        return net

    ## ------------------------------------------------------------ pipeline

    def preprocess(self, image):
        """BGR uint8 image -> RGB float32 HWC array normalized to [-1, 1]."""
        rgb = resize_to_multiple(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return rgb.astype(np.float32) / 127.5 - 1.0

    def run_model(self, input_tensor):
        """(1, 3, H, W) normalized RGB tensor in, raw model output tensor out."""
        mask = self._detect_scratch(input_tensor)
        if self.opt.mask_dilation != 0:
            mask = dilate_mask(mask, self.opt.mask_dilation)

        # paint the scratches white; 255 is 1.0 once normalized to [-1, 1]
        masked = input_tensor * (1 - mask) + mask

        with torch.no_grad():
            generated = self.model.inference(masked, mask)
        return generated.cpu()

    def _detect_scratch(self, input_tensor):
        """Run the detection UNet, returning a binary 0/1 mask of shape (1, 1, H, W)."""
        # luma weights sum to ~1, so this works directly on the normalized tensor
        gray = TF.rgb_to_grayscale(input_tensor)
        gray = gray.to(self.device)
        h, w = input_tensor.shape[2], input_tensor.shape[3]

        with torch.no_grad():
            P = torch.sigmoid(self.detection_net(scale_short_side(gray)))

        P = F.interpolate(P.cpu(), [h, w], mode="nearest")
        return (P >= 0.4).float()

    def inference(self, image):
        """BGR uint8 image in, restored BGR uint8 image out."""
        array = self.preprocess(image)
        input_tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        generated = self.run_model(input_tensor)
        return self.postprocess(generated.squeeze(0).permute(1, 2, 0).numpy())

    def postprocess(self, array):
        """Model output -> BGR uint8, matching save_image(..., normalize=True)."""
        img = (array + 1.0) / 2.0
        low, high = float(img.min()), float(img.max())
        img = (np.clip(img, low, high) - low) / max(high - low, 1e-5)
        rgb = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./test_images/a.png", help="Input image")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--HR", action="store_true")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        sys.exit("Could not read image: %s" % args.input)

    restorer = ImageRestorer(HR=args.HR, device=args.device)
    result = restorer.inference(image)

    root, ext = os.path.splitext(args.input)
    output_path = root + "_out" + ext
    cv2.imwrite(output_path, result)
    print("Saved %s" % output_path)
