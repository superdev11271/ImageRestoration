# onnxruntime-gpu 1.26 needs CUDA 12.x + cuDNN 9; run with `docker run --gpus all`.
# ubuntu24.04 gives python 3.12. For a CPU-only image, swap this base for
# python:3.12-slim and replace onnxruntime-gpu with onnxruntime in requirements.txt.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

# Ubuntu 24.04 marks its system Python externally managed (PEP 668), so install
# into a venv rather than fighting it with --break-system-packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libglib2.0-0t64 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -c "import cv2, numpy, onnxruntime, fastapi, uvicorn, multipart"

COPY onnx_restorer.py server.py ./

# checkpoints/ is not baked in (~360 MB of onnx); mount it at run time:
#   docker run --gpus all -v ./checkpoints:/app/checkpoints -p 8080:8080 <image>
EXPOSE 8080

ENTRYPOINT ["python", "server.py"]
CMD ["--device", "cuda", "--host", "0.0.0.0", "--port", "8080"]
