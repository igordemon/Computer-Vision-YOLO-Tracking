# Autonomous Perception Prototype


## Installation


Python version:
3.11


Create environment:

python -m venv yolo_env


Activate:

.\yolo_env\Scripts\Activate.ps1



Install dependencies:

pip install -r requirements.txt



Install PyTorch:


NVIDIA GPU:

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121


CPU:

pip install torch torchvision



Run:

python yolo_depth_tracking.py