import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from bert_model import BertDeceptionModel
from lie_model import DeceptionModel, demo_rows, load_model, save_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "deception_model.pkl"
BERT_MODEL_PATH = BASE_DIR / "bert_deception_model"
STATIC_DIR = BASE_DIR / "static"


def ensure_model():
    if MODEL_PATH.exists():
        return load_model(MODEL_PATH)
    model = DeceptionModel().fit(demo_rows(), source="built-in demo seed data")
    save_model(model, MODEL_PATH)
    return model


MODEL = ensure_model()
BERT_MODEL = BertDeceptionModel(BERT_MODEL_PATH)


def model_status():
    return {
        "default": "naive_bayes",
        "models": {
            "naive_bayes": {
                "available": True,
                "label": "Naive Bayes",
                "source": MODEL.source,
                "trained_rows": MODEL.trained_rows,
            },
            "bert": {
                "available": BERT_MODEL.available(),
                "label": "BERT",
                "source": BERT_MODEL.source,
                "trained_rows": BERT_MODEL.trained_rows if BERT_MODEL.available() else 0,
            },
        },
    }


def select_model(model_type):
    if model_type in ("", None, "naive_bayes"):
        return MODEL
    if model_type == "bert":
        if not BERT_MODEL.available():
            raise ValueError(
                "BERT model is not available. Add bert_deception_model/ to the project or choose Naive Bayes."
            )
        return BERT_MODEL.load()
    raise ValueError("Unknown model type. Choose naive_bayes or bert.")


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            payload = {"ok": True, "model": MODEL.source, "trained_rows": MODEL.trained_rows}
            payload.update(model_status())
            self.write_json(payload)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = (payload.get("text") or "").strip()
            if len(text) < 20:
                self.write_json({"error": "Please enter at least 20 characters."}, status=400)
                return
            selected_model = select_model(payload.get("model_type", "naive_bayes"))
            result = selected_model.predict(text)
            result["model_type"] = payload.get("model_type", "naive_bayes")
            result["disclaimer"] = (
                "This is a probability-based NLP signal, not proof that a review is truthful or false."
            )
            self.write_json(result)
        except json.JSONDecodeError:
            self.write_json({"error": "Invalid JSON."}, status=400)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            self.write_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self.write_json({"error": str(exc)}, status=500)

    def write_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(host="127.0.0.1", port=8000):
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Lie Detection app running at http://{host}:{port}")
    print(f"Model source: {MODEL.source} ({MODEL.trained_rows} rows)")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the lie detection front end.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run(host=args.host, port=args.port)
