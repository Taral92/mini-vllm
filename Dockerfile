# mini-vllm server image.
#
# CPU (default, small):  docker build -t mini-vllm .
# GPU (needs the NVIDIA container toolkit on the host):
#   docker build -t mini-vllm:gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 .
#   docker run --gpus all -p 8000:8000 mini-vllm:gpu
#
# Weights are downloaded from Hugging Face on first start. Mount a volume on
# /home/app/.cache/huggingface to keep them between runs.
FROM python:3.11-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home app
WORKDIR /app

# torch first, from the chosen index, so it is cached apart from code changes.
RUN pip install --index-url "${TORCH_INDEX}" torch

COPY pyproject.toml README.md run.py ./
COPY engine engine
COPY server server
COPY benchmarks benchmarks
RUN pip install .

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["mini-vllm-serve", "--host", "0.0.0.0", "--port", "8000"]
