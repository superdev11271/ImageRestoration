"""FastAPI server exposing old-photo restoration at POST /api/restore/.

Start with a model name and a device; the models always live in `MODEL_DIR`:

    python server.py --model run_model_fp16.onnx --device cuda

POST an image (multipart field `image`) and get the restored image back as
PNG. Images whose longest side exceeds `--max_side` are downscaled for
inference and scaled back up to their original size. GET /api/health/ for a
liveness check.
"""
import argparse
import os

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from onnx_restorer import OnnxImageRestorer

MODEL_DIR = 'checkpoints'
DEFAULT_MODEL = 'run_model_fp16.onnx'
DEFAULT_DEVICE = 'cuda'
DEFAULT_MAX_SIDE = 1280

app = FastAPI(title='Image Restoration ONNX')
restorer = None
model_info = {}
max_side = DEFAULT_MAX_SIDE


@app.get('/api/health/')
async def health():
    """Report whether the model is loaded and how it is configured."""
    if restorer is None:
        raise HTTPException(status_code=503, detail='model not loaded')
    return {'status': 'ok', **model_info, 'providers': restorer.session.get_providers()}


@app.post('/api/restore/')
async def restore(image: UploadFile = File(...)):
    """Restore the uploaded image and return it as PNG."""
    img = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail='could not decode image')

    h, w = img.shape[:2]
    if max(h, w) > max_side:
        # shrink the long side to max_side, restore, then come back to the original size
        ratio = max_side / max(h, w)
        img = cv2.resize(img, (round(w * ratio), round(h * ratio)), interpolation=cv2.INTER_AREA)
    output = restorer.inference(img)
    # the restorer rounds each side to a multiple of 16, so the result is never
    # quite the size that went in until it is resized back
    output = cv2.resize(output, (w, h), interpolation=cv2.INTER_CUBIC)

    ok, buf = cv2.imencode('.png', output)
    if not ok:
        raise HTTPException(status_code=500, detail='could not encode result')
    return Response(content=buf.tobytes(), media_type='image/png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default=DEFAULT_MODEL,
                        help=f'model file name inside {MODEL_DIR}/')
    parser.add_argument('-d', '--device', type=str, default=DEFAULT_DEVICE, choices=['cuda', 'cpu'])
    parser.add_argument('--max_side', type=int, default=DEFAULT_MAX_SIDE,
                        help='images with a longer side than this are downscaled before inference')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('-p', '--port', type=int, default=8080)
    args = parser.parse_args()

    global restorer, max_side
    model_path = os.path.join(MODEL_DIR, args.model)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    restorer = OnnxImageRestorer(model_path, device=args.device)
    max_side = args.max_side
    model_info.update(model=args.model, device=args.device, max_side=args.max_side)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
