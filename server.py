import base64
import io
import os
import threading
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from diffusers import StableDiffusionPipeline

MODEL_ID = os.getenv(
    "HF_MODEL_ID",
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
)

app = FastAPI(title="YUJ Diffusion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading pipeline '{MODEL_ID}' on {device} ({dtype})...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    dtype=dtype,
    use_safetensors=True,
)
pipe = pipe.to(device)

if hasattr(pipe, "enable_attention_slicing"):
    pipe.enable_attention_slicing()

print("Pipeline loaded successfully.")

gpu_lock = threading.Lock()


class GenerateRequest(BaseModel):
    model_config = {"extra": "ignore"}
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str = ""
    width: int = 768
    height: int = 960
    seed: int = 42
    num_inference_steps: int = 25
    guidance_scale: float = 7.0


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": device}


@app.post("/generate")
def generate(req: GenerateRequest):
    # Scale dimensions so maximum dimension is at most 768 to prevent 6GB VRAM paging/OOM
    max_dim = 768
    orig_w = max(256, req.width)
    orig_h = max(256, req.height)
    scale = min(1.0, max_dim / max(orig_w, orig_h))
    width = int(orig_w * scale)
    height = int(orig_h * scale)
    width = max(256, (width // 8) * 8)
    height = max(256, (height // 8) * 8)

    safe_seed = abs(int(req.seed)) % (2**31 - 1)
    steps = max(1, min(50, int(req.num_inference_steps)))
    guidance = max(1.0, min(20.0, float(req.guidance_scale)))

    try:
        with gpu_lock:
            generator = torch.Generator(device=device).manual_seed(safe_seed)

            with torch.inference_mode():
                result = pipe(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt or None,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                )

            image = result.images[0]
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")

            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "imageBase64": encoded,
            "mimeType": "image/png",
            "meta": {
                "model": MODEL_ID,
                "width": image.width,
                "height": image.height,
                "seed": safe_seed,
            },
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {exc}",
        )
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()


@app.get("/")
def read_root():
    return FileResponse("engine15.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)