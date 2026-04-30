import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from lie_model import DeceptionModel, demo_rows, load_model, save_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "deception_model.pkl"
STATIC_DIR = BASE_DIR / "static"


def ensure_model():
    if MODEL_PATH.exists():
        return load_model(MODEL_PATH)
    model = DeceptionModel().fit(demo_rows(), source="built-in demo seed data")
    save_model(model, MODEL_PATH)
    return model


MODEL = ensure_model()


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.write_json({"ok": True, "model": MODEL.source, "trained_rows": MODEL.trained_rows})
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
            result = MODEL.predict(text)
            result["disclaimer"] = (
                "This is a probability-based NLP signal, not proof that a review is truthful or false."
            )
            self.write_json(result)
        except json.JSONDecodeError:
            self.write_json({"error": "Invalid JSON."}, status=400)
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
