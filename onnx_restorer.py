# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Restoration through the ONNX graph exported by export_onnx.py.

Same pipeline as ImageRestorer - preprocess, model, postprocess - with
onnxruntime in place of the torch half, so this file runs without torch and
without importing anything from inference.py.

The graph takes any input size, but it is exact for the orientation it was
exported for (landscape unless exported with --portrait); portrait images fed
to a landscape graph get a smaller detector input, so scratch masks come out
slightly coarser.
"""

import cv2
import numpy as np
import onnxruntime as ort

# what the graph's input tensor is declared as, and the numpy dtype to feed it
INPUT_DTYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16}


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


class OnnxImageRestorer:
    """Old-photo restoration for a single BGR image, through onnxruntime."""

    def __init__(self, model_path, device="cuda"):
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(model_path, providers=providers)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        if model_input.type not in INPUT_DTYPES:
            raise ValueError("unsupported graph input type %s" % model_input.type)
        self.dtype = INPUT_DTYPES[model_input.type]

    def preprocess(self, image):
        """BGR uint8 image -> RGB float32 HWC array normalized to [-1, 1]."""
        rgb = resize_to_multiple(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return rgb.astype(np.float32) / 127.5 - 1.0

    def run_model(self, batch):
        """(N, 3, H, W) normalized RGB array in, raw model output array out."""
        return self.session.run(None, {self.input_name: batch})[0]

    def inference(self, image):
        """BGR uint8 image in, restored BGR uint8 image out."""
        return self.infer_batch([image])[0]

    def infer_batch(self, images):
        """List of BGR uint8 images in, list of restored BGR uint8 images out.

        The graph's batch axis is dynamic but one call still needs one shape,
        so images are grouped by their preprocessed size: a list of same-size
        images runs in a single call, mixed sizes take one call per size.
        """
        arrays = [self.preprocess(image) for image in images]

        groups = {}
        for index, array in enumerate(arrays):
            groups.setdefault(array.shape, []).append(index)

        results = [None] * len(arrays)
        for indices in groups.values():
            batch = np.stack([arrays[index] for index in indices]).transpose(0, 3, 1, 2)
            # fp16 graphs want half input; preprocess always produces float32
            generated = self.run_model(np.ascontiguousarray(batch, dtype=self.dtype))
            for slot, index in enumerate(indices):
                # postprocess per image: it rescales by the image's own min/max
                results[index] = self.postprocess(generated[slot].transpose(1, 2, 0))
        return results

    def postprocess(self, array):
        """Model output (float32 or float16) -> BGR uint8, matching save_image(normalize=True)."""
        img = (array.astype(np.float32) + 1.0) / 2.0
        low, high = float(img.min()), float(img.max())
        img = (np.clip(img, low, high) - low) / max(high - low, 1e-5)
        rgb = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
