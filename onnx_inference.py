# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Restore a single image with the ONNX graph exported by export_onnx.py."""

import argparse
import os
import sys

import cv2

from onnx_restorer import OnnxImageRestorer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./test_images/a.png", help="Input image")
    parser.add_argument(
        "--model", type=str, default="./checkpoints/run_model.onnx", help="Exported .onnx"
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        sys.exit("Could not read image: %s" % args.input)

    restorer = OnnxImageRestorer(args.model, device=args.device)
    result = restorer.inference(image)

    root, ext = os.path.splitext(args.input)
    output_path = root + "_onnx" + ext
    cv2.imwrite(output_path, result)
    print("Saved %s" % output_path)
