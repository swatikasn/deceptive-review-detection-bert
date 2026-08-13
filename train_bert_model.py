import argparse
import csv
import json
from pathlib import Path


LABEL_AUTHENTIC = 0
LABEL_COMPUTER = 1
DEFAULT_MODEL_NAME = "bert-base-uncased"


def load_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = {name.lower().strip(): name for name in reader.fieldnames or []}
        text_col = next(
            (fieldnames[name] for name in ("text", "text_", "review", "content", "message")
             if name in fieldnames),
            None,
        )
        label_col = next(
            (fieldnames[name] for name in ("label", "target", "class", "category")
             if name in fieldnames),
            None,
        )
        if not text_col or not label_col:
            raise ValueError("CSV must include text/text_/review and label/target columns.")

        for row in reader:
            text = (row.get(text_col) or "").strip()
            raw_label = (row.get(label_col) or "").strip().upper()
            if not text or not raw_label:
                continue
            if raw_label in {"CG", "COMPUTER", "COMPUTER-GENERATED", "AI", "1", "DECEPTIVE", "FAKE"}:
                label = LABEL_COMPUTER
            elif raw_label in {"OR", "ORIGINAL", "HUMAN", "0", "GENUINE", "AUTHENTIC", "REAL"}:
                label = LABEL_AUTHENTIC
            else:
                continue
            rows.append((text, label))
    return rows


class ReviewDataset:
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text, label = self.rows[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        import torch

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train(args):
    try:
        import torch
        from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
        from sklearn.model_selection import train_test_split
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from transformers import BertForSequenceClassification, BertTokenizerFast
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit(
            "Missing BERT training dependencies. Install pandas/scikit-learn/torch/transformers "
            "or run: python3 -m pip install -r requirements-bert.txt"
        ) from exc

    rows = load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]
    if len(rows) < 10:
        raise ValueError("Need at least 10 labeled rows for BERT training.")

    train_rows, test_rows = train_test_split(
        rows,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=[label for _, label in rows],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training rows: {len(train_rows)}")
    print(f"Validation rows: {len(test_rows)}")

    tokenizer = BertTokenizerFast.from_pretrained(args.base_model)
    model = BertForSequenceClassification.from_pretrained(args.base_model, num_labels=2).to(device)

    train_dataset = ReviewDataset(train_rows, tokenizer, args.max_length)
    test_dataset = ReviewDataset(test_rows, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if step % args.log_every == 0:
                print(f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} loss {loss.item():.4f}")

        metrics = evaluate(model, test_loader, device, accuracy_score, roc_auc_score, classification_report)
        avg_loss = total_loss / max(len(train_loader), 1)
        history.append({"epoch": epoch, "loss": avg_loss, **metrics})
        print(
            f"Epoch {epoch}/{args.epochs} complete: "
            f"loss={avg_loss:.4f}, accuracy={metrics['accuracy']:.4f}, roc_auc={metrics['roc_auc']:.4f}"
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "base_model": args.base_model,
        "source_data": str(args.data),
        "training_rows": len(train_rows),
        "validation_rows": len(test_rows),
        "epochs": args.epochs,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "history": history,
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved BERT model to {output_dir}")


def evaluate(model, loader, device, accuracy_score, roc_auc_score, classification_report):
    import torch

    model.eval()
    labels = []
    predictions = []
    probabilities = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            labels.extend(batch["labels"].cpu().numpy().tolist())
            predictions.extend(preds.cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())

    print(
        classification_report(
            labels,
            predictions,
            target_names=["OR (genuine)", "CG (deceptive)"],
            zero_division=0,
        )
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune and export a BERT deception classifier.")
    parser.add_argument("--data", default="fake reviews dataset.csv")
    parser.add_argument("--output", default="bert_deception_model")
    parser.add_argument("--base-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Optional row limit for quick smoke-test training.")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
