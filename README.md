# Image Restoration ONNX

Old-photo restoration through an exported ONNX graph -- as a Python class, a CLI
and a FastAPI server. Send a scratched or faded photo, get a restored one back.

## Layout

| File | Purpose |
| --- | --- |
| [onnx_restorer.py](onnx_restorer.py) | `OnnxImageRestorer` -- the restorer (preprocess -> session -> postprocess) |
| [onnx_inference.py](onnx_inference.py) | CLI: restore one image file |
| [server.py](server.py) | FastAPI server |
| [test.py](test.py) | Smoke-tests the API against an image you pass in |
| `checkpoints/` | `.onnx` weights (not tracked in git) |
| `test_images/` | Sample images used by the smoke checks below (not tracked in git) |

## Install

```bash
pip install -r requirements.txt
```

Python 3.12. `requirements.txt` pins `onnxruntime-gpu`, which needs an NVIDIA driver
with CUDA 12 + cuDNN 9. For a CPU-only machine, install `onnxruntime` instead and
start the server with `--device cpu`.

## Library

```python
import cv2
from onnx_restorer import OnnxImageRestorer

restorer = OnnxImageRestorer('checkpoints/run_model.onnx', device='cuda')
output = restorer.inference(cv2.imread('input.png'))         # BGR uint8 in, BGR uint8 out
outputs = restorer.infer_batch([img1, img2])                 # one session call per distinct size
```

- fp32 and fp16 models are both supported; the input dtype is read from the model.
- `device` is `'cuda'` (default) or `'cpu'`; `'cuda'` falls back to CPU when no CUDA
  execution provider is available.
- Both sides are rounded to a multiple of 16 before inference, so the class returns
  a slightly different size than it was given. The server resizes back for you; a
  direct caller does not get that.
- The graph is exact for the orientation it was exported for. Portrait images fed to
  a landscape graph get a smaller detector input, so scratch masks come out coarser.

## Server

```bash
python server.py
python server.py -m run_model_fp16.onnx -d cpu -p 9000
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `-m`, `--model` | `run_model.onnx` | Restoration model file name |
| `-d`, `--device` | `cuda` | `cuda` or `cpu` |
| `--max_side` | `1920` | Images with a longer side than this are downscaled before inference |
| `--host` | `0.0.0.0` | Bind address |
| `-p`, `--port` | `8080` | Bind port |

The model directory is always `checkpoints/`, so `--model` take a file name, not a path.
Models load once at startup and a missing file fails immediately, so startup takes
a few seconds and the port only opens once every session is ready.

An upload whose longer side exceeds `--max_side` is downscaled to that limit (aspect
ratio kept), restored, then resized back to its **original** dimensions, so oversized images
come back the same size they went in.

## API

Interactive docs at `/docs`.

### `GET /api/health/`

```bash
curl http://127.0.0.1:8080/api/health/
```

```json
{"status": "ok", "model": "run_model.onnx", "device": "cuda", "max_side": 1920,
 "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}
```

`providers` comes from the live session, so it shows whether CUDA actually engaged or
fell back to CPU. `503` before loading has finished.

### `POST /api/restore/`

Request -- `multipart/form-data`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `image` | file | required | Image to restore |

Response -- the restored image as raw `image/png` bytes at the input's original
dimensions. `400` if the upload cannot be decoded as an image, `500` if the result
cannot be encoded.

```bash
curl -X POST -F "image=@test_images/a.png" http://127.0.0.1:8080/api/restore/ -o test_images/a_out.png
```

## Docker

Base image `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04`, so run it with `--gpus all`.
`checkpoints/` is not baked into the image (~540 MB of onnx) -- mount it at run time.

```bash
docker build -t imagerestoration-server .
docker run --gpus all -p 8080:8080 -v ./checkpoints:/app/checkpoints imagerestoration-server
```

Server flags pass straight through the entrypoint:

```bash
docker run --gpus all -p 8080:8080 -v ./checkpoints:/app/checkpoints imagerestoration-server -m run_model_fp16.onnx --max_side 1280
```

Omit `--gpus all` and pass `-d cpu` to run on CPU. On Windows use an absolute path for
the mount, e.g. `-v "E:\Project\phoenix\servers\ImageRestoration\checkpoints:/app/checkpoints"`.

To bake the models into the image instead, drop `checkpoints/` from `.dockerignore` and add
`COPY checkpoints/*.onnx ./checkpoints/` to the Dockerfile.

### GPU notes

`--gpus all` requires Docker Desktop's **WSL2 backend** (Settings -> General -> *Use the
WSL 2 based engine*). On the Hyper-V backend the container has no NVIDIA driver and
fails with `nvidia-container-cli: initialization error: load library failed:
libnvidia-ml.so.1`.

## Test

`test.py` takes the input image path as its argument and checks the health route, a
round-trip of that image, and that a non-image upload is rejected. It exits non-zero
if any check fails. Each line carries the response time for that request, so you can
see what the model actually costs per call. It needs `requests`, which is deliberately
kept out of `requirements.txt` so it stays out of the Docker image.

```bash
pip install requests
python server.py --device cpu &
python test.py test_images/a.png
```

```
[PASS] health -- 6ms -- {'status': 'ok', 'model': 'run_model.onnx', ...}
[PASS] restore -- 2089ms -- (450, 298, 3) -> (450, 298, 3)
       wrote test_images/a_out.png
[PASS] rejects a non-image -- 3ms -- 400 {"detail":"could not decode image"}
3/3 checks passed
```

Those times are from a CPU run on this sample; `--device cuda` and the fp16 graph
give very different numbers.

| Flag | Default | Meaning |
| --- | --- | --- |
| `image` | required | Path to the image to send |
| `--url` | `http://127.0.0.1:8080` | Base url of the running server |
| `-o`, `--output` | `test_images/a_out.png` | Where to write the returned png; defaults to the input path with `_out.png` in place of its extension |

Point it at another host or a container with `--url`:

```bash
python test.py test_images/a.png --url http://127.0.0.1:9000
```

### With curl

The same three checks by hand, without `test.py` or `requests`:

```bash
curl -s http://127.0.0.1:8080/api/health/
# {"status": "ok", ...}

curl -s -X POST -F "image=@test_images/a.png" http://127.0.0.1:8080/api/restore/ -o test_images/a_out.png
python -c "import cv2; print(cv2.imread('test_images/a.png').shape, '->', cv2.imread('test_images/a_out.png').shape)"
# (450, 298, 3) -> (450, 298, 3)

curl -s -o /dev/null -w '%{http_code}\n' -X POST -F "image=@README.md" http://127.0.0.1:8080/api/restore/
# 400
```

### Inference time

`OnnxImageRestorer.inference()` on `test_images/a.png` (450x298), mean of 10 runs
after 3 warmup runs, on an RTX 4070:

| Model | Device | Provider | Mean | Std | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| `run_model.onnx` | cuda | CUDAExecutionProvider | 70.7ms | 0.5ms | 69.9ms | 71.7ms |
| `run_model.onnx` | cpu | CPUExecutionProvider | 1867.0ms | 43.6ms | 1821.2ms | 1961.4ms |
| `run_model_fp16.onnx` | cuda | CUDAExecutionProvider | 47.2ms | 1.0ms | 46.2ms | 49.9ms |
| `run_model_fp16.onnx` | cpu | CPUExecutionProvider | 2579.6ms | 50.4ms | 2500.7ms | 2658.5ms |

`--device cuda` is the difference that matters: ~26x on the fp32 graph. fp16 is then
the faster graph on GPU (1.5x over fp32) but the slower one on CPU, which has no
native half kernels, so the casts are pure overhead there. These are model times
only -- the endpoint adds PNG decode and encode on top.

## Notes

- ONNX sessions are created at startup and shared. FastAPI runs the endpoint in a
  threadpool, so concurrent requests are correct but throughput is bounded by the
  session a request runs on.
- The fp16 graph is selected by name: `--model run_model_fp16.onnx`. It halves the
  weights on disk; the class reads the dtype from the graph and feeds it half input.
- The restorer rounds each side to a multiple of 16, so the graph never sees the exact
  input size. The endpoint resizes the result back, which is why the response matches
  the upload while `onnx_inference.py` writes the rounded size.
- `util/` holds no modules, only a `__pycache__`; nothing here imports it.
