FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

COPY . .

CMD ["python3", "CV_Project/yolo_depth_tracking.py"]
