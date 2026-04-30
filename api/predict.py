import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lie_model import load_model  # noqa: E402


MODEL = load_model(ROOT / "models" / "deception_model.pkl")


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = (payload.get("text") or "").strip()
            if len(text) < 20:
                self._json({"error": "Please enter at least 20 characters."}, status=400)
                return

            result = MODEL.predict(text)
            result["disclaimer"] = (
                "This is a probability-based NLP signal, not proof that a review is truthful or false."
            )
            self._json(result)
        except json.JSONDecodeError:
            self._json({"error": "Invalid JSON."}, status=400)
        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
