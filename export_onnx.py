# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Export ImageRestorer.run_model to ONNX.

run_model is the whole torch half of the pipeline (scratch detection, mask
dilation, masking and the restoration generators), so it exports as a single
graph. preprocess / postprocess stay in cv2 / numpy on the caller's side.

By default the graph is static: it takes exactly one 1 x 3 x height x width
input. With --dynamic, batch, height and width stay symbolic and one export
serves any input size; height and width are then exported as 16 * dim, because
both networks downsample by 16, which is what preprocess produces.

Either way one thing is fixed at export time: which side scale_short_side treats as the
shorter one. torch.export resolves that test from the sample input, so the
graph is exact for that orientation and scales portrait input by 256 / height
instead of 256 / width - aspect ratio preserved, detector input just smaller.
A static export takes its orientation from --height / --width; a dynamic one
is landscape unless you pass --portrait. Feed portrait images to a portrait
graph, or transpose them, if that matters.

Export goes through torch.export (dynamo=True); the TorchScript exporter
resolves the size arithmetic too and bakes the sample sizes into the graph.
"""

import argparse
import os

import torch
from torch.export import Dim

from inference import ImageRestorer

# (batch, long side, short side), transposed for a --portrait graph
VERIFY_SHAPES = [(1, 256, 256), (2, 320, 256), (1, 480, 208), (3, 176, 144)]


class RunModelWrapper(torch.nn.Module):
    """nn.Module front-end for ImageRestorer.run_model, for the exporter."""

    def __init__(self, restorer):
        super(RunModelWrapper, self).__init__()
        self.detection_net = restorer.detection_net
        self.netG_A = restorer.model.netG_A
        self.netG_B = restorer.model.netG_B
        self.mapping_net = restorer.model.mapping_net
        self.restorer = restorer

    def forward(self, x):
        return self.restorer.run_model(x)


def export(output_path, height, width, opset, HR, portrait, dynamic):
    if height % 16 or width % 16:
        raise ValueError("height and width must be multiples of 16, got %dx%d" % (height, width))
    if dynamic and portrait:
        height, width = width, height

    # export on CPU: the graph has no device notion, and run_model's .cuda() /
    # .cpu() hops would otherwise be traced as device transfers
    restorer = ImageRestorer(HR=HR, device="cpu")
    wrapper = RunModelWrapper(restorer).eval()

    if dynamic:
        dynamic_shapes = {
            "x": {
                0: Dim("batch", min=1, max=64),
                2: 16 * Dim("h", min=2, max=512),
                3: 16 * Dim("w", min=2, max=512),
            }
        }
        # batch 2 and a non-square sample: torch.export specializes size-1
        # dimensions, and equal dimensions could collapse into one symbol
        sample = torch.randn(2, 3, height, width)
    else:
        dynamic_shapes = None
        sample = torch.randn(1, 3, height, width)

    program = torch.onnx.export(
        wrapper,
        (sample,),
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
    )
    program.save(output_path)
    print("Saved %s" % output_path)
    return restorer


def verify(output_path, restorer, portrait, dynamic, height, width):
    """Compare the exported graph against the torch model at the shapes it accepts."""
    import numpy as np
    import onnxruntime as ort

    if dynamic:
        shapes = [
            (batch, long_side, short_side) if portrait else (batch, short_side, long_side)
            for batch, long_side, short_side in VERIFY_SHAPES
        ]
    else:
        shapes = [(1, height, width)]

    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    for batch, height, width in shapes:
        x = torch.randn(batch, 3, height, width)
        with torch.no_grad():
            expected = restorer.run_model(x).numpy()
        actual = session.run(None, {"input": x.numpy()})[0]
        print(
            "%dx3x%dx%d max abs diff torch vs onnxruntime: %.3e"
            % (batch, height, width, np.abs(expected - actual).max())
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=str, default="./checkpoints/run_model.onnx", help="Output .onnx path"
    )
    parser.add_argument("--height", type=int, default=256, help="Input height, multiple of 16")
    parser.add_argument("--width", type=int, default=320, help="Input width, multiple of 16")
    parser.add_argument(
        "--dynamic", action="store_true", help="Export dynamic batch/height/width instead of static"
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--HR", action="store_true")
    parser.add_argument(
        "--portrait", action="store_true", help="With --dynamic, export for height > width input"
    )
    parser.add_argument("--verify", action="store_true", help="Check the graph with onnxruntime")
    args = parser.parse_args()

    directory = os.path.dirname(os.path.abspath(args.output))
    if not os.path.isdir(directory):
        os.makedirs(directory)

    restorer = export(
        args.output, args.height, args.width, args.opset, args.HR, args.portrait, args.dynamic
    )
    if args.verify:
        verify(args.output, restorer, args.portrait, args.dynamic, args.height, args.width)
