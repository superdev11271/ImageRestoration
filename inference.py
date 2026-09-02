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

from restoration_net import RestorationNet
from scratch_detector import ScratchDetector


class ImageRestorer:
    """Old-photo restoration for a single BGR image."""

    def __init__(self, HR=False, GPU=0):
        self.HR = HR
        self.GPU = GPU

        self.detector = ScratchDetector(GPU=GPU)
        # the mapping model differs per mode, so both variants are loaded
        self.restoration_quality = RestorationNet(False, HR=HR, GPU=GPU)
        self.restoration_scratch = RestorationNet(True, HR=HR, GPU=GPU)

    def inference(self, image, with_scratch=False):
        """BGR uint8 image in, restored BGR uint8 image out."""
        if with_scratch:
            return self.restoration_scratch.restore(image, self.detector.detect(image))
        return self.restoration_quality.restore(image)


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

    restorer = ImageRestorer(HR=args.HR, GPU=args.GPU)
    result = restorer.inference(image, with_scratch=args.with_scratch)

    root, ext = os.path.splitext(args.input)
    output_path = root + "_out" + ext
    cv2.imwrite(output_path, result)
    print("Saved %s" % output_path)
