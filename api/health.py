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


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = {
            "ok": True,
            "model": MODEL.source,
            "trained_rows": MODEL.trained_rows,
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
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
