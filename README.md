# YUJ Diffusion Studio & Backend

Local GPU-accelerated Stable Diffusion backend powered by FastAPI and Hugging Face Diffusers, paired with the YUJ Studio visual asset generator.

## Features
- **GPU Acceleration**: Utilizes CUDA (NVIDIA GeForce RTX 3050 Laptop GPU or similar) with FP16 precision.
- **Fast Generation**: Generates high-quality assets in ~8 seconds with attention slicing and memory management.
- **Interactive Studio**: Full canvas and asset composer served directly from `engine15.html`.
- **Flexible Diffuser**: Easily switch Hugging Face models using environment variables or configuration.

## Requirements
- Python 3.10+
- NVIDIA GPU with CUDA support
- Dependencies in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

## Quick Start
1. Double-click **`run_backend.bat`** (or run `python server.py`).
2. Open your browser to:
   ```
   http://127.0.0.1:8000/
   ```
3. Enter your prompt in the **Visual Concept** field and click **Generate Asset**.

## How to Change the Diffuser Model
You can change the underlying model anytime without changing code:

### Option 1: Via Environment Variable
```powershell
$env:HF_MODEL_ID = "runwayml/stable-diffusion-v1-5"  # or any diffusers-compatible model
python server.py
```

### Option 2: In `server.py`
Update `MODEL_ID` in `server.py`:
```python
MODEL_ID = os.getenv(
    "HF_MODEL_ID",
    "your-chosen-model-id-here",
)
```

## API Endpoints
- `GET /` — Serves the frontend web studio (`engine15.html`).
- `GET /health` — Status and active GPU/model info.
- `POST /generate` — Generates image from JSON payload:
  ```json
  {
    "prompt": "Taj Mahal, photorealistic, cinematic lighting",
    "negative_prompt": "blurry, low quality",
    "width": 512,
    "height": 640,
    "seed": 42,
    "num_inference_steps": 20
  }
  ```
