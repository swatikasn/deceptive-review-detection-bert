from pathlib import Path

from lie_model import LABEL_AUTHENTIC, LABEL_COMPUTER, MODEL_VERSION, extract_indicators


BERT_MODEL_VERSION = f"{MODEL_VERSION}-bert"


class BertDeceptionModel:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.source = "bert-base-uncased fine-tuned model"
        self.trained_rows = "from BERT training notebook"
        self._model = None
        self._tokenizer = None
        self._torch = None

    def available(self):
        return self.model_dir.exists()

    def load(self):
        if not self.available():
            raise FileNotFoundError(
                f"BERT model folder not found at {self.model_dir}. "
                "Train/export it from the notebook as bert_deception_model first."
            )

        try:
            import torch
            from transformers import BertForSequenceClassification, BertTokenizerFast
        except ImportError as exc:
            raise RuntimeError(
                "BERT dependencies are not installed. Install torch and transformers to use this model."
            ) from exc

        if self._model is None or self._tokenizer is None:
            self._torch = torch
            self._tokenizer = BertTokenizerFast.from_pretrained(str(self.model_dir))
            self._model = BertForSequenceClassification.from_pretrained(str(self.model_dir))
            self._model.eval()
        return self

    def predict_proba(self, text):
        self.load()
        encoded = self._tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            logits = self._model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            ).logits
            return self._torch.softmax(logits, dim=1)[0, 1].item()

    def predict(self, text, threshold=0.5):
        score = self.predict_proba(text)
        return {
            "deception_score": round(score, 4),
            "prediction": "Likely computer-generated" if score >= threshold else "Likely authentic",
            "label": LABEL_COMPUTER if score >= threshold else LABEL_AUTHENTIC,
            "confidence": round(max(score, 1 - score), 4),
            "key_indicators": extract_indicators(text),
            "model_source": self.source,
            "trained_rows": self.trained_rows,
            "model_version": BERT_MODEL_VERSION,
        }
