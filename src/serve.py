import io
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).parent))

try:
    from model import get_model
except ImportError:
    from src.model import get_model


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

NORM_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
NORM_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocesses PIL image to normalized CIFAR-10 tensor (1, 3, 32, 32).
    """
    resized = image.resize((32, 32), Image.Resampling.BILINEAR)
    img_np = np.array(resized, dtype=np.float32) / 255.0
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]
    tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    tensor = (tensor - NORM_MEAN) / NORM_STD
    return tensor.unsqueeze(0)  # (1, 3, 32, 32)


app = FastAPI(
    title="CIFAR-10 Image Classification Serving API",
    description="FastAPI service for serving PyTorch image classification models.",
    version="1.0.0",
)

model: Optional[torch.nn.Module] = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_metadata: Dict[str, Any] = {}


def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_checkpoint(checkpoint_path: Optional[str] = None) -> bool:
    global model, model_metadata
    try:
        config = load_config()
        arch = config.get("model", {}).get("architecture", "simple_cnn")
        num_classes = config.get("model", {}).get("num_classes", 10)

        if not checkpoint_path:
            ckpt_dir = config.get("output", {}).get("checkpoint_dir", "./checkpoints")
            model_name = config.get("output", {}).get("model_name", "best_model.pt")
            checkpoint_path = os.getenv("MODEL_PATH", f"{ckpt_dir}/{model_name}")

        net = get_model(architecture=arch, num_classes=num_classes)
        ckpt_file = Path(checkpoint_path)

        if ckpt_file.exists():
            checkpoint = torch.load(ckpt_file, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                net.load_state_dict(checkpoint["model_state_dict"])
                model_metadata = {
                    "checkpoint_path": str(ckpt_file),
                    "epoch": checkpoint.get("epoch"),
                    "val_loss": checkpoint.get("val_loss"),
                    "val_accuracy": checkpoint.get("val_accuracy"),
                }
            elif isinstance(checkpoint, dict):
                net.load_state_dict(checkpoint)
                model_metadata = {"checkpoint_path": str(ckpt_file)}
            print(f"Successfully loaded checkpoint from: {ckpt_file}")
        else:
            print(f"Warning: Checkpoint not found at '{checkpoint_path}'. Model initialized with default weights.")
            model_metadata = {"checkpoint_path": None, "warning": "Initialized without preloaded checkpoint"}

        net.to(device)
        net.eval()
        model = net
        return True
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        model = None
        model_metadata = {"error": str(e)}
        return False


@app.on_event("startup")
def startup_event():
    load_model_checkpoint()


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "CIFAR-10 PyTorch Serving API",
        "health_check": "/health",
        "predict_endpoint": "/predict",
    }


@app.get("/health", tags=["Health"])
def health_check():
    if model is not None:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "healthy",
                "model_loaded": True,
                "device": str(device),
                "metadata": model_metadata,
            },
        )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "unhealthy",
            "model_loaded": False,
            "error": model_metadata.get("error", "Model is not loaded"),
        },
    )


@app.post("/predict", tags=["Inference"])
async def predict(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    upload = file or image
    if upload is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image uploaded. Please provide an image file using the 'file' or 'image' form field.",
        )

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable.",
        )

    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{upload.content_type}'. Please upload an image file.",
        )

    try:
        contents = await upload.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process image file: {str(e)}",
        )

    try:
        input_tensor = preprocess_image(img).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze(0)

        prob_list = probabilities.cpu().tolist()
        pred_idx = int(torch.argmax(probabilities).item())
        confidence = float(prob_list[pred_idx])

        class_probabilities = {
            CIFAR10_CLASSES[i] if i < len(CIFAR10_CLASSES) else f"class_{i}": round(prob, 6)
            for i, prob in enumerate(prob_list)
        }
        pred_class = CIFAR10_CLASSES[pred_idx] if pred_idx < len(CIFAR10_CLASSES) else f"class_{pred_idx}"

        return {
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "confidence": round(confidence, 6),
            "probabilities": class_probabilities,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("serve:app", host=host, port=port, reload=False)
