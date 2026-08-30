# MLOps PyTorch Pipeline: CIFAR-10 Training & Serving

A modular, production-ready **MLOps PyTorch Pipeline** for training, evaluating, containerizing, and serving deep learning models on the CIFAR-10 dataset using **FastAPI**, **Docker**, and **Kubernetes** orchestration.

---

## Project Overview

This repository provides an end-to-end Machine Learning pipeline featuring:
- **Modular Model Architectures**: Configurable PyTorch models including `SimpleCNN`, `ResNet18`, and `ResNet34` tailored for CIFAR-10.
- **Config-Driven Training**: YAML-based hyperparameter configuration (`configs/training_config.yaml`) with early stopping, dynamic metric logging (JSON-formatted), and automated checkpointing.
- **High-Performance Inference Server**: Asynchronous FastAPI service (`src/serve.py`) providing `/health` check probes and `/predict` multipart image inference endpoints.
- **Optimized Multi-Stage Dockerization**: Lightweight, secure container builds for training (`Dockerfile.train`) and serving (`Dockerfile.serve`) running as non-root users (`appuser`) with built-in health checks.
- **Full Traceability & Execution Logs**: Comprehensive execution logs and visual screenshots stored in the `logs/` directory.

---

## Repository Structure

```tree
mlops-pytorch-pipeline/
├── configs/
│   └── training_config.yaml      # Hyperparameters, paths & training configuration
├── docker/
│   ├── Dockerfile.train          # Multi-stage Dockerfile for model training
│   └── Dockerfile.serve          # Multi-stage Dockerfile for FastAPI serving (non-root)
├── logs/                         # 📁 Execution logs & validation evidence
│   ├── Part_B_Logs.txt           # Complete CLI logs for Docker build, train & serve
│   ├── Screenshot #1.png         # Image build & training start screenshot
│   ├── Screenshot #2.png         # Checkpointing & training progress screenshot
│   ├── Screenshot #3.png         # Docker serving build & startup screenshot
│   ├── Screenshot #3.1.png       # Serving container deployment verification
│   ├── Screenshot #4.png         # Health check and predict requests logs
│   ├── Screenshot #5.png         # Inference output response screenshot
│   └── partb_screenshot_1.png    # End-to-end execution summary screenshot
├── requirements/
│   ├── train.txt                 # Dependencies for training (torch, torchvision, pyyaml, etc.)
│   ├── serve.txt                 # Dependencies for serving (fastapi, uvicorn, pillow, etc.)
│   └── serve2.txt                # Additional serving dependencies
├── src/
│   ├── dataset.py                # CIFAR-10 data loaders & augmentation transforms
│   ├── model.py                  # PyTorch model definitions (SimpleCNN, ResNet18, ResNet34)
│   ├── train.py                  # Training pipeline with validation & early stopping
│   └── serve.py                  # FastAPI inference REST API server
├── k8s/                          # Kubernetes manifests for container orchestration
│   ├── namespace.yaml            # Dedicated namespace
│   ├── configmap.yaml            # Configuration mappings
│   ├── training-job.yaml         # Training job specification
│   ├── serving-deployment.yaml   # Inference deployment configuration
│   ├── serving-service.yaml      # ClusterIP / NodePort service routing
│   └── hpa.yaml                  # Horizontal Pod Autoscaler definition
├── checkpoints/                  # Saved model weights (e.g. best_model.pt)
├── data/                         # Downloaded CIFAR-10 datasets
├── Docker Cmds.yml               # Quick reference cheat-sheet for Docker commands
├── test_image.png                # Sample image for testing the prediction endpoint
└── README.md                     # Project documentation
```

---

## Logs & Evidence Location

All execution records, terminal outputs, and verification screenshots are centrally located in the [`logs/`](file:///g:/MLOps_Ass3/mlops-pytorch-pipeline/logs) directory:

| File / Asset | Description |
| :--- | :--- |
| [`logs/Part_B_Logs.txt`](file:///g:/MLOps_Ass3/mlops-pytorch-pipeline/logs/Part_B_Logs.txt) | Raw text log output covering Docker image builds, multi-epoch training logs, model checkpoint saves (`best_model.pt`), server startup, `/health` probes, and `/predict` curl test requests. |
| `logs/Screenshot #1.png` | Verification of Docker training image build. |
| `logs/Screenshot #2.png` | Model training progression and loss/accuracy convergence. |
| `logs/Screenshot #3.png` & `Screenshot #3.1.png` | FastAPI serving container build and initialization. |
| `logs/Screenshot #4.png` | Server health checks (`GET /health`) and inference call logs (`POST /predict`). |
| `logs/Screenshot #5.png` | Final curl output with classification predictions and class probabilities. |
| `logs/partb_screenshot_1.png` | High-level execution snapshot. |
| **Kubernetes Logs** | *Deployment and validation in progress — yet to be updated.* |

> **Note:** Kubernetes deployment and validation are currently in progress and are yet to be updated.

---

## Configuration

The pipeline is configured via [`configs/training_config.yaml`](file:///g:/MLOps_Ass3/mlops-pytorch-pipeline/configs/training_config.yaml):

```yaml
model:
  architecture: "simple_cnn"   # Options: "simple_cnn", "resnet18", "resnet34"
  num_classes: 10

data:
  data_dir: "./data"

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 20
  early_stopping_patience: 5

output:
  checkpoint_dir: "./checkpoints"
  model_name: "best_model.pt"
```

---

## Quickstart & Execution Guide

### 1. Local Python Setup

```bash
# Clone the repository
git clone https://github.com/RajaGanapathyM/mlops-pytorch-pipeline.git
cd mlops-pytorch-pipeline

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements/train.txt
pip install -r requirements/serve.txt
```

#### Run Training Locally:
```bash
python src/train.py
```

#### Run Serving Locally:
```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8080
```

---

### 2. Docker Containerized Workflow

#### Step 1: Build the Training Image
```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
```

#### Step 2: Run Training with Volume Mounts
Persist dataset downloads and checkpoint weights on your host machine:
```bash
# On Linux/macOS / PowerShell:
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/checkpoints:/app/checkpoints" mlops-train:v1

# On Windows Command Prompt:
docker run --rm -v "%cd%/data:/app/data" -v "%cd%/checkpoints:/app/checkpoints" mlops-train:v1
```

#### Step 3: Build the Serving Image
```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

#### Step 4: Run Serving Container
Mount checkpoints into the inference server and bind port `8080`:
```bash
# On Linux/macOS / PowerShell:
docker run --rm -p 8080:8080 -v "$(pwd)/checkpoints:/app/checkpoints" mlops-serve:v1

# On Windows Command Prompt:
docker run --rm -p 8080:8080 -v "%cd%/checkpoints:/app/checkpoints" mlops-serve:v1
```

---

## REST API Reference

The serving API exposes the following endpoints on `http://localhost:8080`:

### 1. Root Info
- **Endpoint**: `GET /`
- **Description**: Returns basic service information and links to health and predict endpoints.

### 2. Health Check
- **Endpoint**: `GET /health`
- **Description**: Validates that the model is loaded and ready for inference.
- **Example Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "metadata": {
    "checkpoint_path": "checkpoints/best_model.pt",
    "epoch": 19,
    "val_loss": 0.5336,
    "val_accuracy": 0.8184
  }
}
```

### 3. Predict Endpoint
- **Endpoint**: `POST /predict`
- **Request Format**: Multipart Form Data (`file` or `image` field)
- **Curl Example**:
```bash
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```
- **Example Response**:
```json
{
  "predicted_class": "airplane",
  "predicted_index": 0,
  "confidence": 0.99992,
  "probabilities": {
    "airplane": 0.99992,
    "automobile": 0.0,
    "bird": 0.000054,
    "cat": 0.000005,
    "deer": 0.000004,
    "dog": 0.0,
    "frog": 0.0,
    "horse": 0.0,
    "ship": 0.000003,
    "truck": 0.000014
  }
}
```

---

## Kubernetes Deployment (k8s)

Manifests for cluster deployment are organized under [`k8s/`](file:///g:/MLOps_Ass3/mlops-pytorch-pipeline/k8s):
- `namespace.yaml`: Defines isolated namespace for workloads.
- `configmap.yaml`: Injects hyperparameters and runtime configurations.
- `training-job.yaml`: Batch Job for executing model training in-cluster.
- `serving-deployment.yaml`: Replicated Pod deployment for FastAPI inference.
- `serving-service.yaml`: Load balanced service endpoint for internal/external access.
- `hpa.yaml`: Horizontal Pod Autoscaler targeting CPU/Memory thresholds.

> **Status:** Cluster deployment execution and validation are currently in progress and will be updated once testing is completed.

---

## Supported CIFAR-10 Classes

| Index | Class | Index | Class |
|:---:|:---|:---:|:---|
| `0` | **airplane** | `5` | **dog** |
| `1` | **automobile** | `6` | **frog** |
| `2` | **bird** | `7` | **horse** |
| `3` | **cat** | `8` | **ship** |
| `4` | **deer** | `9` | **truck** |

---

## License

This project is created for educational and practical MLOps pipeline demonstrations.