# Serving image: web UI + REST API on port 8000, GPU required.
#
#   docker build -t manga-assistant .
#   docker run --gpus all -p 8000:8000 \
#     -v %cd%/models:/app/models -v %cd%/data:/app/data -v %cd%/outputs:/app/outputs \
#     -v %USERPROFILE%/.cache/huggingface:/cache/hf  manga-assistant
#
# Weights and data are mounted, not baked in: they are versioned by DVC, the image by git.
# CUDA 12.8 because the RTX 50 series (Blackwell, sm_120) needs it.
# Training (kohya) and the dataset build (Magi venv) stay outside the container.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
RUN pip install --no-cache-dir --no-deps -e .

ENV HF_HOME=/cache/hf
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
