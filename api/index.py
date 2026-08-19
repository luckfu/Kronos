"""Thin HTTP adapter for the model-only serverless service."""

from flask import Flask, jsonify, request

from serverless.service import RequestError, predict

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "kronos-inference"})


@app.post("/predict")
def predict_route():
    try:
        return jsonify(predict(request.get_json(silent=True)))
    except RequestError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Kronos inference failed")
        return jsonify({"error": f"inference failed: {exc}"}), 500
