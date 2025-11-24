FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Paquetes de sistema para OpenCV y V4L2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    v4l2loopback-utils \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos TODO el código
COPY . /app

# 🔥 Limpiamos cualquier numpy/opencv viejo e instalamos un stack sano
RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y numpy opencv-python opencv-python-headless || true && \
    pip install --no-cache-dir \
        "numpy<2.0.0" \
        "opencv-python==4.9.0.80" \
        "fastapi==0.111.0" \
        "uvicorn[standard]==0.30.0" \
        "PyYAML==6.0.2"

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
