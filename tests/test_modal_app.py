from pathlib import Path


def test_modal_app_uses_v6_checkpoint_and_model_only_boundary():
    deploy_dir = Path(__file__).parents[1] / "deploy" / "modal"
    source = deploy_dir / "modal_app.py"
    text = source.read_text()

    assert 'MODEL_REPO_ID = "luckfu/Kronos-A-Share-Forecast"' in text
    assert "force_build=True" in text
    assert "serverless.service" in text
    assert "snapshot_download" in text
    assert "KRONOS_MODEL_ID" in text
    assert "payload: dict = Body(...)" in text
    assert "request: Request" not in text
    assert (deploy_dir / "requirements.txt").is_file()
    assert (deploy_dir / "README.md").is_file()
    assert (deploy_dir / "curl_test.sh").is_file()
