"""Modal deployment for the Kronos Small model-only inference API."""

import os
from pathlib import Path

import modal
from fastapi import Body, FastAPI, Header
from fastapi.responses import JSONResponse


DEPLOY_DIR = Path(__file__).resolve().parent
MODEL_REPO_ID = "luckfu/Kronos-small-0.1-Cosine-C2-Best"
TOKENIZER_REPO_ID = "NeoQuasar/Kronos-Tokenizer-base"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
REMOTE_REPO_PATH = "/opt/kronos/models/small_0_1_cosine_c2_best"
REMOTE_MODEL_PATH = REMOTE_REPO_PATH
REMOTE_TOKENIZER_PATH = "/opt/kronos/models/kronos_tokenizer_base"
TOKENIZER_ID = os.getenv("KRONOS_TOKENIZER_ID", REMOTE_TOKENIZER_PATH)
SECRET_NAME = os.getenv("MODAL_KRONOS_SECRET_NAME", "")


app = modal.App("kronos-beta-v1-2-inference")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(DEPLOY_DIR / "requirements.txt"))
    .run_commands(
        "python -c \"from modelscope import snapshot_download; "
        f"snapshot_download('{MODEL_REPO_ID}', local_dir='{REMOTE_REPO_PATH}', "
        "allow_patterns=['config.json', 'model.safetensors'])\"",
        "python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id='{TOKENIZER_REPO_ID}', revision='{TOKENIZER_REVISION}', "
        f"local_dir='{REMOTE_TOKENIZER_PATH}', allow_patterns=['config.json', 'model.safetensors'])\"",
        # Re-read the release snapshot on every deployment so a replaced remote
        # artifact cannot remain hidden in a cached Modal image layer.
        force_build=True,
    )
    .env({
        "KRONOS_MODEL_ID": REMOTE_MODEL_PATH,
        "KRONOS_TOKENIZER_ID": TOKENIZER_ID,
        "MODAL_MODEL_REPO_ID": MODEL_REPO_ID,
    })
    .add_local_python_source("model", "serverless")
)
secrets = [modal.Secret.from_name(SECRET_NAME)] if SECRET_NAME else []


def _check_api_key(authorization: str | None) -> None:
    expected = os.getenv("KRONOS_API_KEY")
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="invalid API key")


@app.function(
    image=image,
    secrets=secrets,
    gpu="T4",
    scaledown_window=10,
    timeout=180,
)
@modal.asgi_app()
def web():
    from serverless.service import RequestError, predict, predict_batch

    api = FastAPI(title="Kronos Small 0.1 Inference API", version="small-0.1")

    @api.get("/health")
    def health():
        return {
            "status": "ok",
            "service": "kronos-beta-v1-2-inference",
            "model": MODEL_REPO_ID,
            "release": "small-0.1-cosine-c2",
            "checkpoint": "Segment@179",
        }

    @api.post("/predict")
    async def predict_route(
        payload: dict = Body(...),
        authorization: str | None = Header(default=None),
    ):
        _check_api_key(authorization)
        try:
            return predict(payload)
        except RequestError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse(
                {"error": f"inference failed: {exc}"}, status_code=500
            )

    @api.post("/predict-batch")
    async def predict_batch_route(
        payload: dict = Body(...), authorization: str | None = Header(default=None),
    ):
        _check_api_key(authorization)
        try:
            return predict_batch(payload)
        except RequestError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": f"batch inference failed: {exc}"}, status_code=500)

    return api


if __name__ == "__main__":
    print("Run: modal deploy deploy/modal/modal_app.py")
