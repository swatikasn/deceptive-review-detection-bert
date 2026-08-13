import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bert_model import BertDeceptionModel  # noqa: E402
from lie_model import load_model  # noqa: E402


MODEL = load_model(ROOT / "models" / "deception_model.pkl")
BERT_MODEL = BertDeceptionModel(ROOT / "bert_deception_model")


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


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = (payload.get("text") or "").strip()
            if len(text) < 20:
                self._json({"error": "Please enter at least 20 characters."}, status=400)
                return

            selected_model = select_model(payload.get("model_type", "naive_bayes"))
            result = selected_model.predict(text)
            result["model_type"] = payload.get("model_type", "naive_bayes")
            result["disclaimer"] = (
                "This is a probability-based NLP signal, not proof that a review is truthful or false."
            )
            self._json(result)
        except json.JSONDecodeError:
            self._json({"error": "Invalid JSON."}, status=400)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            self._json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
