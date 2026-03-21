# NovaDev Cloud — Free tier notebook container
# Pre-installs Jupyter + common ML libs + Ollama for LLM installer

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip python3.11-venv \
    curl wget git nano \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf python3.11 /usr/bin/python && ln -sf pip3 /usr/bin/pip

# ── Python packages ───────────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Ollama (for one-click LLM install) ────────────────────────────────────────
RUN curl -fsSL https://ollama.com/install.sh | sh

# ── Jupyter config ────────────────────────────────────────────────────────────
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_origin = '*'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py

WORKDIR /workspace

# ── Startup script ────────────────────────────────────────────────────────────
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8888   
# Jupyter notebook port
EXPOSE 11434  
# Ollama LLM API port

CMD ["/start.sh"]

# ─────────────────────────────────────────────────────────────────────────────
# requirements.txt content (create as separate file):
#   jupyter==1.0.0
#   jupyterlab==4.1.0
#   torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121
#   torchvision torchaudio
#   numpy pandas scikit-learn matplotlib seaborn
#   transformers datasets accelerate peft
#   huggingface_hub
#   xgboost lightgbm
#   opencv-python-headless pillow
#   fastapi uvicorn httpx aiofiles
