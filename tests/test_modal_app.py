from pathlib import Path


def test_modal_app_uses_small_checkpoint_and_model_only_boundary():
    deploy_dir = Path(__file__).parents[1] / "deploy" / "modal"
    source = deploy_dir / "modal_app.py"
    text = source.read_text()

    assert 'MODEL_REPO_ID = "luckfu/Kronos-small-0.1-Cosine-C2-Best"' in text
    assert 'TOKENIZER_REPO_ID = "NeoQuasar/Kronos-Tokenizer-base"' in text
    assert "REMOTE_MODEL_PATH = REMOTE_REPO_PATH" in text
    assert 'REMOTE_TOKENIZER_PATH = "/opt/kronos/models/kronos_tokenizer_base"' in text
    assert 'modal.App("kronos-beta-v1-2-inference")' in text
    assert "force_build=True" in text
    assert "allow_patterns=['config.json', 'model.safetensors']" in text
    assert "huggingface_hub import snapshot_download" in text
    assert "serverless.service" in text
    assert "snapshot_download" in text
    assert "KRONOS_MODEL_ID" in text
    assert "payload: dict = Body(...)" in text
    assert "request: Request" not in text
    assert (deploy_dir / "requirements.txt").is_file()
    assert (deploy_dir / "README.md").is_file()
    assert (deploy_dir / "curl_test.sh").is_file()
